# models/embeddings.py
import io
import pdfplumber
import re
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# small persistent cache directory (optional)
CACHE_DIR = Path(".cache_fast_store")
CACHE_DIR.mkdir(exist_ok=True)


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def safe_extract_pdf_text(pdf_bytes: bytes) -> str:
    """Robust PDF extraction that works on Streamlit Cloud."""
    text = ""

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    ptxt = page.extract_text() or ""
                    text += ptxt + "\n"
                except:
                    pass
    except:
        pass

    # Clean and normalize
    text = re.sub(r"\s+", " ", text)
    text = text.replace("•", " ").replace("▪", " ").replace("●", " ")
    text = re.sub(r"[^a-zA-Z0-9.,:/\-() ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes quickly. Single-pass.
    """
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        parts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            parts.append(t)
        return "\n".join(parts)
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 300) -> List[str]:
    """
    Chunk by characters. Default chunk_size tuned for small docs.
    """
    if not text:
        return []

    chunks = []
    s = 0
    n = len(text)
    while s < n:
        e = min(s + chunk_size, n)
        chunk = text[s:e].strip()
        if chunk:
            chunks.append(chunk)
        s = e - overlap
    return chunks


def build_tfidf_store_from_bytes(
    file_bytes: bytes,
    filename_hint: Optional[str] = None,
    chunk_size: int = 3000,
    overlap: int = 300,
    persist: bool = False,
) -> Dict[str, Any]:
    """
    Build a TF-IDF store (chunks + vectorizer + matrix) from PDF bytes.
    """
    key = sha256_bytes(file_bytes) if filename_hint is None else filename_hint
    # if persisted and exists, load
    if persist:
        p = CACHE_DIR / f"{key}.pkl"
        if p.exists():
            try:
                with open(p, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                pass

    text = extract_text_from_pdf_bytes(file_bytes)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        store = {"chunks": [], "vectorizer": None, "matrix": None, "key": key}
        if persist:
            with open(CACHE_DIR / f"{key}.pkl", "wb") as fh:
                pickle.dump(store, fh)
        return store

    vect = TfidfVectorizer(stop_words="english", max_features=20000)
    mat = vect.fit_transform(chunks)

    store = {"chunks": chunks, "vectorizer": vect, "matrix": mat, "key": key}
    if persist:
        try:
            with open(CACHE_DIR / f"{key}.pkl", "wb") as fh:
                pickle.dump(store, fh)
        except Exception:
            pass

    return store


def retrieve_relevant_chunks(
    query: str,
    vector_store: Dict[str, Any],
    top_k: int = 3,
    min_score: float = -1.0
) -> List[str]:
    """
    Robust retrieval that works with various vector_store shapes:
      - {'chunks', 'vectorizer', 'vectors' (np.ndarray)}   <- app.py style
      - {'chunks', 'vectorizer', 'matrix' (sparse)}        <- older style (sparse matrix)
      - {'chunks', 'embeddings' (list[list[float]])}       <- precomputed embeddings

    Returns up to top_k chunk strings. If min_score >= 0, only returns chunks with score >= min_score.
    """
    if not vector_store:
        return []

    chunks = vector_store.get("chunks", []) or []
    if not chunks:
        return []

    vect = vector_store.get("vectorizer", None)
    matrix = vector_store.get("matrix", None)  # could be sparse
    vectors = vector_store.get("vectors", None)  # dense numpy array
    embeddings = vector_store.get("embeddings", None)  # list-of-lists fallback

    # Case A: sparse matrix + vectorizer (fast)
    if vect is not None and matrix is not None:
        try:
            qv = vect.transform([query])
            sims = linear_kernel(qv, matrix).flatten()  # linear_kernel on TF-IDF ≈ cosine
            idxs = np.argsort(sims)[::-1][:top_k]
            results = []
            for i in idxs:
                if min_score < 0 or sims[int(i)] >= min_score:
                    results.append(chunks[int(i)])
            return results
        except Exception:
            pass

    # Case B: dense vectors + vectorizer
    if vect is not None and vectors is not None:
        try:
            qv = vect.transform([query]).toarray().astype(np.float32)
            vecs = np.asarray(vectors, dtype=np.float32)
            if qv.shape[1] == vecs.shape[1]:
                sims = cosine_similarity(qv.reshape(1, -1), vecs).flatten()
                idxs = np.argsort(-sims)[:top_k]
                results = []
                for i in idxs:
                    if min_score < 0 or sims[int(i)] >= min_score:
                        results.append(chunks[int(i)])
                return results
        except Exception:
            pass

    # Case C: precomputed embeddings (list-of-lists or np.array)
    if embeddings is not None:
        try:
            arr = np.asarray(embeddings, dtype=np.float32)
            if vect is not None:
                qv = vect.transform([query]).toarray().astype(np.float32)
                if qv.shape[1] == arr.shape[1]:
                    sims = cosine_similarity(qv.reshape(1, -1), arr).flatten()
                    idxs = np.argsort(-sims)[:top_k]
                    results = []
                    for i in idxs:
                        if min_score < 0 or sims[int(i)] >= min_score:
                            results.append(chunks[int(i)])
                    return results
            # If we don't have a way to embed the query, skip to substring fallback
        except Exception:
            pass

    # Last resort: substring/token scoring fallback
    qtokens = [t for t in query.lower().split() if len(t) > 2]
    if not qtokens:
        return []

    scores = []
    for c in chunks:
        cl = c.lower()
        score = sum(cl.count(tok) for tok in qtokens)
        scores.append(float(score))
    scores = np.array(scores, dtype=np.float32)

    if scores.sum() == 0:
        qset = set(query.lower().split())
        scores = np.array([len(qset.intersection(set(c.lower().split()))) for c in chunks], dtype=np.float32)

    if scores.sum() == 0:
        return []

    idxs = np.argsort(-scores)[:top_k]
    selected = [chunks[int(i)] for i in idxs if min_score < 0 or scores[int(i)] >= min_score]
    return selected
