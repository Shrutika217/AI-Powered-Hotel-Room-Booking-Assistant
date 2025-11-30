# app.py (fixed)
import streamlit as st
import hashlib
import os
import sys
import time
import re
import random
from datetime import datetime, timedelta
from typing import List, Callable, Optional, Tuple, Dict

# Optional: load .env for local env vars
try:
  from dotenv import load_dotenv
  load_dotenv()
except Exception:
  pass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity

# make project modules importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Project-specific imports (lazy imports used where appropriate to avoid circular imports)
from config.config import settings
from models.embeddings import extract_text_from_pdf_bytes
from utils.chat_logic import handle_user_message
from utils.database import init_db, list_bookings

from streamlit.components.v1 import html as st_html

def scroll_to_bottom():
  # tiny JS to scroll page to bottom — height 0 so no UI change
  st_html("<script>window.scrollTo(0,document.body.scrollHeight);</script>", height=0)

def safe_append_assistant_reply(reply: str):
  """
  Append assistant reply to st.session_state.messages only if the
  same assistant reply is not already the last assistant message.
  This handles cases where handle_user_message() may itself append.
  """
  msgs = st.session_state.get("messages", [])
  if msgs:
      last = msgs[-1]
      if last.get("role") == "assistant" and last.get("content") == reply:
          return
  st.session_state.messages.append({"role": "assistant", "content": reply})

# -------------------------
# Helpers: TF-IDF & chunking
# -------------------------
def file_sha256(b: bytes) -> str:
  h = hashlib.sha256()
  h.update(b)
  return h.hexdigest()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
  chunks = []
  L = len(text)
  if L == 0:
      return chunks
  start = 0
  while start < L:
      end = start + chunk_size
      chunk = text[start:end]
      chunks.append(chunk.strip())
      start += chunk_size - overlap
  return chunks


def batch_transform_to_dense(vectorizer: TfidfVectorizer, texts: List[str], batch_size: int = 128,
                           progress_cb: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
  n = len(texts)
  if n == 0:
      # No texts -> return empty (0, n_features) array for consistency if possible
      # but we don't have n_features; return explicit empty 2-D array
      return np.zeros((0, 0), dtype=np.float32)
  mats = []
  for i in range(0, n, batch_size):
      batch = texts[i: i + batch_size]
      mat = vectorizer.transform(batch)  # sparse matrix
      mats.append(mat)
      if progress_cb:
          processed = min(i + batch_size, n)
          progress_cb(processed, n)
  # Stack sparse matrices and convert once
  from scipy.sparse import vstack as sp_vstack
  stacked = sp_vstack(mats)
  arr = stacked.toarray().astype(np.float32)
  return arr


def build_local_tfidf_store_from_text(text: str, chunk_size=1200, overlap=200,
                                    progress_vec_cb: Optional[Callable[[int, int], None]] = None):
  chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
  if not chunks:
      return {"chunks": [], "vectorizer": None, "vectors": None}
  vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, ngram_range=(1, 2), dtype=np.float32)
  # fit - wrap in try to catch odd tokenization errors
  try:
      vectorizer.fit(chunks)
  except Exception as e:
      # fitting failed
      print("[TFIDF DEBUG] vectorizer.fit failed:", e)
      return {"chunks": chunks, "vectorizer": None, "vectors": None}
  vectors = batch_transform_to_dense(vectorizer, chunks, batch_size=128, progress_cb=progress_vec_cb)
  # if vectors is empty 0x0, convert to None to signal empty
  if vectors is None or vectors.size == 0:
      vectors = None
  return {"chunks": chunks, "vectorizer": vectorizer, "vectors": vectors}


def get_top_k_from_store(store: dict, query: str, k: int = 3):
  # returns (idxs, sims) -- both lists
  if not store:
      return [], []
  vectors = store.get("vectors")
  vectorizer = store.get("vectorizer")
  if vectors is None or vectorizer is None:
      return [], []
  # ensure we have a 2D vectors array
  if vectors.ndim != 2:
      return [], []
  try:
      qv = vectorizer.transform([query]).toarray().astype(np.float32)
      if qv.size == 0 or vectors.shape[1] != qv.shape[1]:
          # shape mismatch
          return [], []
      sims = cosine_similarity(qv.reshape(1, -1), vectors).flatten()
      idxs = np.argsort(-sims)[:k]
      return list(idxs), list(sims[idxs])
  except Exception as e:
      print("[RAG DEBUG] get_top_k_from_store failed:", e)
      return [], []

# -------------------------------------------
# STRICT HOTEL NAME EXTRACTION (RECOMMENDED)
# -------------------------------------------
# STRICT HOTEL NAME EXTRACTION (FINAL VERSION)
# -------------------------------------------

HOTEL_SUFFIX = r"(Hotel|Resort|Inn|Lodge|Suites|Suite|Villa|Palace|Residence|Residences|Resorts)"

# Match ONLY names that END with the suffix (no trailing text allowed)
HOTEL_NAME_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9&'’\-\s]{1,80}?\s+" + HOTEL_SUFFIX + r")\b",
    flags=re.IGNORECASE
)

# Remove "Brochure", “Brochure -”, “Brochure of”, etc.
_LEADING_NOISE_RE = re.compile(
    r"^\s*(brochure(\sof)?\s*[:\-–—]?)",
    flags=re.IGNORECASE
)

# Remove marketing lines like “Thank you…”, “Welcome to…”, “Introducing…”
_MARKETING_PREFIX_RE = re.compile(
    r"^(thank you.*|thanks.*|welcome to.*|welcome.*|introducing.*|presenting.*)$",
    flags=re.IGNORECASE
)

def _clean_hotel_name(text: str) -> str:
    """Clean marketing/brochure noise and normalize spacing."""
    text = _MARKETING_PREFIX_RE.sub("", text).strip()
    text = _LEADING_NOISE_RE.sub("", text).strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,.:;-")

def clean_filename_title(name: str) -> str:
    """
    Clean fallback title from filename.
    Removes 'Brochure', numbers, hyphens, underscores, file extensions.
    """
    # Remove extension
    name = re.sub(r"\.(pdf|docx?|txt)$", "", name, flags=re.IGNORECASE)

    # Replace underscores/hyphens with spaces
    name = name.replace("_", " ").replace("-", " ")

    # Remove 'Brochure', 'Brochure of', etc.
    name = re.sub(r"\bbrochure\b", "", name, flags=re.IGNORECASE)

    # Collapse spaces
    name = re.sub(r"\s{2,}", " ", name).strip()

    return name.strip(" ,.-")


def extract_hotel_name_strict(text: str) -> Optional[str]:
    """Extract clean hotel name ending with Hotel/Resort/Suite/etc."""
    if not text:
        return None

    head = text[:8000]

    # Remove marketing phrases at top of PDF
    head = _MARKETING_PREFIX_RE.sub("", head)

    # -------------------------------------------
    # Primary extraction: exact match ending with suffix
    # -------------------------------------------
    for m in HOTEL_NAME_PATTERN.finditer(head):
        raw = m.group(1).strip()
        clean = _clean_hotel_name(raw)

        # Reject if containing URLs/contact junk
        if re.search(r"(www|http|email|@|phone|tel)", clean, re.IGNORECASE):
            continue

        # Must look like a proper title
        if not clean[0].isupper():
            continue

        # No more than 8 words (avoids sentences)
        if len(clean.split()) > 8:
            continue

        return clean

    # -------------------------------------------
    # Fallback: manual scan of early lines
    # -------------------------------------------
    for line in head.splitlines()[:120]:
        line = line.strip()
        if not line:
            continue

        # Only lines containing a hotel suffix are of interest
        if re.search(HOTEL_SUFFIX, line, re.IGNORECASE):
            cleaned = _clean_hotel_name(line)

            # Only keep portion up to hotel suffix
            match = HOTEL_NAME_PATTERN.search(cleaned)
            if match:
                cleaned = match.group(1).strip()

            if re.search(r"(www|http|email|@|phone|tel)", cleaned, re.IGNORECASE):
                continue

            if cleaned and cleaned[0].isupper() and len(cleaned.split()) <= 8:
                return cleaned

    return None


# -------------------------
# Very strict room extraction: only from explicit "Room Types" / "Rooms Available" sections
# -------------------------
# -------------------------
# Very strict room extraction (expanded heading patterns + debug)
# -------------------------
ROOM_SECTION_HEADINGS_RE = re.compile(
  r'^(?:\s*(?:room types|rooms available|rooms:|room types:|our rooms|room categories|room types and descriptions|rooms & suites|rooms\/suites|rooms and suites|rooms & room types))\s*$',
  flags=re.IGNORECASE | re.MULTILINE
)

def extract_room_types_strict(text: str, max_items: int = 12, debug: bool = False) -> List[str]:
  """
  Collect lines immediately following explicit headings such as 'Room Types', 'Rooms & Suites', 'Rooms Available', etc.
  Stop at blank line or next heading. Return cleaned short names only.

  If debug=True, returns a tuple: (room_list, debug_info_dict)
  """
  debug_info = {"found_heading_at": None, "raw_candidates": [], "accepted": [], "rejected": []}
  if not text:
      return ([] if not debug else ([], debug_info))

  lines = [ln.rstrip() for ln in text.splitlines()]
  normalized = [ln.strip() for ln in lines]
  candidates = []
  N = len(normalized)
  i = 0
  while i < N:
      if ROOM_SECTION_HEADINGS_RE.match(normalized[i]):
          debug_info["found_heading_at"] = i
          # collect following lines until blank or next top-level heading
          j = i + 1
          while j < N:
              nxt = normalized[j]
              if not nxt:
                  break
              # stop if looks like the next section heading (short and ends with :)
              if re.match(r'^[A-Za-z &\/]{1,80}[:\-]?$' , nxt) and len(nxt.split()) <= 6 and nxt.endswith(":"):
                  break

              # normalize bullets/leading numbers and remove common separators at start
              candidate_raw = re.sub(r'^[\-\•\*\d\.\)\s]+', '', nxt).strip()
              debug_info["raw_candidates"].append((j, candidate_raw))

              # skip lines that clearly contain contact/price/URLs
              if re.search(r'\b(www\.|http:|https:|@|phone|tel:|₹|\$|€|per night|per month)\b', candidate_raw, flags=re.IGNORECASE):
                  debug_info["rejected"].append((j, candidate_raw, "contains contact/price/url"))
                  j += 1
                  continue

              # Accept if it contains a room keyword OR looks like a short title-cased phrase (up to 8 tokens)
              has_room_keyword = bool(re.search(r'\b(Room|Suite|Villa|Apartment|Studio|Deluxe|Executive|Premium|Luxury|Ocean)\b', candidate_raw, flags=re.IGNORECASE))
              is_short_title = len(candidate_raw.split()) <= 8 and candidate_raw and candidate_raw[0].isupper()

              # avoid long sentences: if it has multiple sentences, skip
              if len(candidate_raw) > 2 and (candidate_raw.count('.') >= 1 or candidate_raw.count('?') >= 1):
                  debug_info["rejected"].append((j, candidate_raw, "looks like sentence"))
                  j += 1
                  continue

              if (has_room_keyword or is_short_title) and len(candidate_raw) <= 120:
                  # cleanup trailing punctuation
                  clean = re.sub(r'[\:\-\–\—\s]+$', '', candidate_raw).strip(" ,.:;-")
                  # final sanity: avoid generic lines that include "amenit", "policy", "contact"
                  if re.search(r'\b(amenit|policy|contact|rate|price|per night|capacity)\b', clean, flags=re.IGNORECASE):
                      debug_info["rejected"].append((j, candidate_raw, "contains generic terms"))
                      j += 1
                      continue
                  candidates.append(clean)
                  debug_info["accepted"].append((j, clean))
              else:
                  debug_info["rejected"].append((j, candidate_raw, "no room keyword or not title-cased/too long"))
              j += 1
          break
      i += 1

  # dedupe & limit
  unique = []
  seen = set()
  for c in candidates:
      key = c.lower()
      if key not in seen:
          seen.add(key)
          unique.append(" ".join(c.split()))
      if len(unique) >= max_items:
          break

  if debug:
      return unique, debug_info
  return unique

# -------------------------
# Booking validation & utilities
# -------------------------
NAME_REGEX = re.compile(r"^[A-Za-z][A-Za-z'`.\-]{1,}(\s+[A-Za-z][A-Za-z'`.\-]{1,})+$")
EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_REGEX = re.compile(r"^[6-9]\d{9}$") 


def looks_like_name(text: str) -> bool:
  txt = text.strip()
  if len(txt) < 3 or len(txt) > 60:
      return False
  if " " not in txt:
      return False
  if any(ch.isdigit() for ch in txt):
      return False
  return bool(NAME_REGEX.match(txt))

def extract_email(text: str) -> Optional[str]:
  m = EMAIL_REGEX.search(text)
  return m.group(0) if m else None

def looks_like_phone(text: str) -> Optional[str]:
   """Return sanitized 10-digit phone number if valid, else None."""
   digits = re.sub(r"[^\d]", "", text)
   if PHONE_REGEX.match(digits):
       return digits  # return only the 10 digits
   return None


def generate_verification_code() -> str:
  return f"{random.randint(100000, 999999)}"


VERIFICATION_TTL = timedelta(minutes=15)


def apply_single_change_and_summary(session_state: dict, field: str, newval: str) -> str:
  """
  Apply a single-field change to the current booking_state in session_state,
  and return a friendly assistant message that echoes the change and shows
  the updated booking summary and next actions.

  - field: normalized lower-case name (e.g. 'name','email','phone','room','checkin','checkout','guests')
  - newval: the user-supplied new value (string)
  """
  bs = session_state.get("booking_state")
  if not bs:
      return "No active booking to change."

  slots = bs.setdefault("slots", {})
  field = field.lower().strip()
  
  def make_summary_lines():
      return (
          f"- Full name: {slots.get('name')}\n"
          f"- Email: {slots.get('email')}\n"
          f"- Phone: {slots.get('phone')}\n"
          f"- Room type: {slots.get('room_type')}\n"
          f"- Check-in: {slots.get('check_in_date')}\n"
          f"- Check-out: {slots.get('check_out_date')}\n"
          f"- Guests: {slots.get('num_guests')}\n"
      )

  # NAME
  if field == "name":
      slots["name"] = newval.strip()
      session_state["booking_state"] = bs
      summary = make_summary_lines()
      return (
          f"Name set to {slots['name']}. Here is the updated booking summary:\n\n"
          f"{summary}\n"
          "If the details are correct, reply with Yes to confirm and save the booking. "
          "To change a field, reply with change <field> (for example: 'change email')."
      )

  # EMAIL (requires verification first)
  if field == "email":
      em = extract_email(newval)
      if not em:
          return "That does not look like a valid email address. Please provide a valid email."

      # set new email but mark as unverified and require code entry
      slots["email"] = em
      bs["verified"] = False
      code = generate_verification_code()
      bs["verification_code"] = code
      bs["verification_sent_at"] = datetime.utcnow().isoformat()
      # set step to verify_code so booking handler will expect the code next
      bs["step"] = "verify_code"
      session_state["booking_state"] = bs

      ok, msg = send_verification_email(em, code)
      if ok:
          return (
              f"Email set to {em}. A verification code was sent to {em}. "
              "Please enter the 6-digit code here to complete the change. "
              "After verification you'll see the updated booking summary and can reply Yes to confirm or change <field> to edit further."
          )
      else:
          return (
              f"Email set to {em} (failed to send code: {msg}). "
              "Please provide an alternative email or try again."
          )

  # PHONE
  if field in ("phone", "phone number"):
      digits = re.sub(r"[^\d]", "", newval)
      if not digits:
          return "Phone number invalid. Please provide digits only."
      if len(digits) >= 7 and len(digits) <= 15:
          full = f"{getattr(settings, 'DEFAULT_COUNTRY_CODE', '+91')} {digits}"
          slots["phone"] = full
          session_state["booking_state"] = bs
          summary = make_summary_lines()
          return (
              f"Phone set to {full}. Here is the updated booking summary:\n\n"
              f"{summary}\n"
              "If the details are correct, reply with Yes to confirm and save the booking. "
              "To change a field, reply with change <field>."
          )
      else:
          return "Phone number looks invalid (unexpected length). Please provide the correct number."

  # ROOM (accept numeric index or exact name)
  if field in ("room", "room_type", "room type"):
      opts = session_state.get("room_options") or []
      nv = newval.strip()
      if nv.isdigit():
          idx = int(nv)
          if 1 <= idx <= len(opts):
              slots["room_type"] = opts[idx - 1]
              session_state["booking_state"] = bs
              summary = make_summary_lines()
              return (
                  f"Room set to {slots['room_type']}. Here is the updated booking summary:\n\n"
                  f"{summary}\n"
                  "If the details are correct, reply with Yes to confirm and save the booking. "
                  "To change a field, reply with change <field>."
              )
          else:
              return f"Invalid room index. Please provide a number between 1 and {len(opts)}."
      else:
          matches = [o for o in opts if o.lower() == nv.lower()]
          if matches:
              slots["room_type"] = matches[0]
          else:
              slots["room_type"] = nv
          session_state["booking_state"] = bs
          summary = make_summary_lines()
          return (
              f"Room set to {slots['room_type']}. Here is the updated booking summary:\n\n"
              f"{summary}\n"
              "If the details are correct, reply with Yes to confirm and save the booking. "
              "To change a field, reply with change <field>."
          )

  # CHECK-IN
  if field in ("checkin", "check_in_date", "check-in"):
      try:
          dt = datetime.strptime(newval.strip(), "%Y-%m-%d").date()
          if dt < datetime.utcnow().date():
              return "Check-in cannot be in the past. Please provide a future date YYYY-MM-DD."
          slots["check_in_date"] = dt.strftime("%Y-%m-%d")
          session_state["booking_state"] = bs
          summary = make_summary_lines()
          return (
              f"Check-in set for {slots['check_in_date']}. Here is the updated booking summary:\n\n"
              f"{summary}\n"
              "If the details are correct, reply with Yes to confirm and save the booking. "
              "To change a field, reply with change <field>."
          )
      except Exception:
          return "Invalid date format. Please provide YYYY-MM-DD."




  # CHECK-OUT
  if field in ("checkout", "check_out_date", "check-out"):
      try:
          dt = datetime.strptime(newval.strip(), "%Y-%m-%d").date()
          checkin = slots.get("check_in_date")
          if checkin:
              try:
                  ci = datetime.strptime(checkin, "%Y-%m-%d").date()
                  if dt < ci:
                      return "Check-out cannot be before check-in. Please choose a valid date."
              except Exception:
                  pass
          slots["check_out_date"] = dt.strftime("%Y-%m-%d")
          session_state["booking_state"] = bs
          summary = make_summary_lines()
          return (
              f"Check-out set for {slots['check_out_date']}. Here is the updated booking summary:\n\n"
              f"{summary}\n"
              "If the details are correct, reply with Yes to confirm and save the booking. "
              "To change a field, reply with change <field>."
          )
      except Exception:
          return "Invalid date format. Please provide YYYY-MM-DD."


  # GUESTS
  if field in ("guests", "num_guests", "number of guests"):
      digits = re.sub(r"[^\d]", "", newval)
      if not digits:
          return "Please enter a positive whole number for guests."
      n = int(digits)
      if n < 1:
          return "Number of guests must be at least 1."
      slots["num_guests"] = n
      session_state["booking_state"] = bs
      summary = make_summary_lines()
      return (
          f"Guests set to {n}. Here is the updated booking summary:\n\n"
          f"{summary}\n"
          "If the details are correct, reply with Yes to confirm and save the booking. "
          "To change a field, reply with change <field>."
      )

  return "I did not recognise that field. You can change: name, email, phone, room, checkin, checkout, guests."

# -------------------------
# BOOKING FLOW WRAPPER
# -------------------------
def handle_booking_step(user_text: str, session_state: dict) -> str:
  """
  Centralized booking flow processor with immediate echo prompts.
  Returns a string reply to show to the user.
  """
  bs = session_state.get("booking_state")
  if not bs:
      return "No active booking session."

  slots = bs.setdefault("slots", {})
  step = bs.get("step", "ask_name")

  # ---------- NEW: if user previously requested "change <field>" and is now providing the new value ----------
  pending_field = bs.get("pending_change_field")
  if pending_field:
      # apply change using helper and clear pending flag
      reply = apply_single_change_and_summary(session_state, pending_field, user_text)
      bs.pop("pending_change_field", None)
      session_state["booking_state"] = bs
      return reply
  # -------------------------------------------------------------------------------------------------------

  # Step: ask_name
  if step == "ask_name":
      if looks_like_name(user_text):
          slots["name"] = user_text.strip()
          bs["step"] = "ask_email"
          session_state["booking_state"] = bs
          return f"Welcome {slots['name']}! Please provide your email id for verification."
      else:
          return "That doesn't look like a valid full name. Please provide first and last name (e.g., 'Shrutika Gupta')."

  # Step: ask_email
  if step == "ask_email":
      email = extract_email(user_text)
      if email:
          slots["email"] = email
          code = generate_verification_code()
          bs["verification_code"] = code
          bs["verification_sent_at"] = datetime.utcnow().isoformat()
          bs["step"] = "verify_code"
          session_state["booking_state"] = bs


          ok, msg = send_verification_email(email, code)
          if ok:
              return f"Email ID set to {email}. A verification code has been sent to {email}. Please enter the 6-digit code here."
          else:
              return f"Unable to send verification email to {email}. Reason: {msg}. Please provide an alternative email or try again."
      else:
          return "That doesn't look like a valid email address. Please provide a valid email (for example: user@example.com)."

  # Step: verify_code
  if step == "verify_code":
      code_try = user_text.strip()
      stored = bs.get("verification_code")
      sent_at_iso = bs.get("verification_sent_at")
      expired = False
      if sent_at_iso:
          try:
              sent_at = datetime.fromisoformat(sent_at_iso)
              if datetime.utcnow() - sent_at > VERIFICATION_TTL:
                  expired = True
          except Exception:
              expired = False

      if expired:
          bs["verification_code"] = None
          bs["verification_sent_at"] = None
          bs["step"] = "ask_email"
          session_state["booking_state"] = bs
          return "The verification code has expired. Please re-enter your email address to receive a new code."


      if code_try.lower() == "resend":
        new_code = generate_verification_code()
        bs["verification_code"] = new_code
        bs["verification_sent_at"] = datetime.utcnow().isoformat()
        session_state["booking_state"] = bs
        ok2, m2 = send_verification_email(bs["slots"].get("email"), new_code)
        if ok2:
            return "A new verification code was sent to your email."
        else:
            return f"Failed to send new verification code: {m2}"

      if stored and code_try == str(stored):
          bs["verified"] = True

          # If this verification happened during a CHANGE-EMAIL flow,
          # show updated booking summary instead of continuing the normal flow.
          if bs.get("pending_change_field") == "email":
              summary = (
                  "Your email has been successfully updated and verified.\n\n"
                  "Please review your updated booking details:\n\n"
                  f"- Full name: {slots.get('name')}\n"
                  f"- Email: {slots.get('email')}\n"
                  f"- Phone: {slots.get('phone')}\n"
                  f"- Room type: {slots.get('room_type')}\n"
                  f"- Check-in: {slots.get('check_in_date')}\n"
                  f"- Check-out: {slots.get('check_out_date')}\n"
                  f"- Guests: {slots.get('num_guests')}\n\n"
                  "If everything looks correct, please reply **Yes** to confirm your booking.\n"
                  "If you need to change a field again, reply with **change <field>**."
              )

              bs["step"] = "summary"
              bs["pending_change_field"] = None
              session_state["booking_state"] = bs
              return summary

          # Otherwise continue normal flow
          bs["step"] = "ask_phone"
          session_state["booking_state"] = bs
          return f"Email verified for {slots.get('email')}. Please provide your 10-digit contact number."

  # Step: ask_phone
  if step == "ask_phone":
      phone = looks_like_phone(user_text)
      if phone:
          full_phone = f"+91 {phone}"
          slots["phone"] = full_phone

          # Move to room selection
          bs["step"] = "ask_room_type"
          session_state["booking_state"] = bs

          # Fetch extracted rooms
          room_opts = session_state.get("room_options") or []

          if room_opts:
              numbered = "\n".join([f"{i+1}. {r}" for i, r in enumerate(room_opts)])
              return (
                  f"Contact number set to {full_phone}.\n\n"
                  "Please select your preferred room type by entering its number:\n\n"
                  f"{numbered}\n\n"
                  "Type the number of your choice."
              )
          else:
              return (
                  f"Contact number set to {full_phone}.\n"
                  "Please type your desired room type (e.g., Deluxe, Standard, Suite)."
              )

      return "Please enter a valid **10-digit Indian mobile number** (starting with 6, 7, 8, or 9)."

  
  # Step: ask_room_type
  if step == "ask_room_type":

    # Fetch extracted room types
    room_opts = session_state.get("room_options") or []

    # If room types exist, show the numbered list
    if room_opts:
        numbered = "\n".join([f"{i+1}. {r}" for i, r in enumerate(room_opts)])

        # If user enters a NUMBER → map to room name
        if user_text.strip().isdigit():
            idx = int(user_text.strip())
            if 1 <= idx <= len(room_opts):
                chosen = room_opts[idx - 1]
                slots["room_type"] = chosen
                bs["step"] = "ask_checkin"
                session_state["booking_state"] = bs
                return f"Room type set to **{chosen}**. Please type **OK** to proceed to selecting your check-in date."
            else:
                return f"Invalid selection. Please choose a number between 1 and {len(room_opts)}."

        # FIRST TIME ASKING → SHOW LIST
        return (
            "Please select your preferred room type by entering its number:\n\n"
            f"{numbered}\n\n"
            "Type the number of your choice."
        )

    # If NO room types extracted → fallback free-text flow
    else:
        rt = user_text.strip()
        if len(rt) < 2:
            return "Please provide a valid room type name."

        slots["room_type"] = rt
        bs["step"] = "ask_checkin"
        session_state["booking_state"] = bs
        return f"Room type set to {slots['room_type']}. Please type **OK** to proceed to selecting your check-in date."

  # Step: ask_checkin
  if step == "ask_checkin":
      try:
          dt = datetime.strptime(user_text.strip(), "%Y-%m-%d").date()
          if dt < datetime.utcnow().date():
              raise ValueError("past")
          slots["check_in_date"] = dt.strftime("%Y-%m-%d")
          bs["step"] = "ask_checkout"
          session_state["booking_state"] = bs
          return f"Check-in date set for {slots['check_in_date']}. Please enter your checkout date in YYYY-MM-DD format OR 'same' for a single night."
      except Exception:
          return "Invalid date. Please enter the date in YYYY-MM-DD format and ensure it is today or in the future."

  # Step: ask_checkout
  if step == "ask_checkout":
      t = user_text.strip().lower()
      if t in ("same", "same day", "one night"):
          slots["check_out_date"] = slots.get("check_in_date")
          bs["step"] = "ask_guests"
          session_state["booking_state"] = bs
          return f"Check-out date set for {slots['check_out_date']} (same as check-in). How many guests will stay? (enter a number)"
      else:
          try:
              dt = datetime.strptime(user_text.strip(), "%Y-%m-%d").date()
              if dt < datetime.strptime(slots.get("check_in_date"), "%Y-%m-%d").date():
                  raise ValueError("checkout before checkin")
              slots["check_out_date"] = dt.strftime("%Y-%m-%d")
              bs["step"] = "ask_guests"
              session_state["booking_state"] = bs
              return f"Check-out date set for {slots['check_out_date']}. How many guests will stay? (enter a number)"
          except Exception:
              return "Invalid check-out date. Please enter YYYY-MM-DD or reply 'same'."

  # Step: ask_guests
  if step == "ask_guests":
      try:
          n = int(re.sub(r"[^\d]", "", user_text.strip()))
          if n <= 0:
              raise ValueError()
          slots["num_guests"] = n
          bs["step"] = "summary"
          session_state["booking_state"] = bs
      except Exception:
          return "Please enter a valid number of guests (for example: 2)."

      # Build summary (echoing again before confirmation)
      summary = (
          "Please review your booking details:\n\n"
          f"- Full name: {slots.get('name')}\n"
          f"- Email: {slots.get('email')}\n"
          f"- Phone: {slots.get('phone')}\n"
          f"- Room type: {slots.get('room_type')}\n"
          f"- Check-in: {slots.get('check_in_date')}\n"
          f"- Check-out: {slots.get('check_out_date')}\n"
          f"- Guests: {slots.get('num_guests')}\n\n"
          "If all details are correct, please reply with **Yes** to confirm and save the booking. "
          "If you need to change any detail, reply with **change <field>** (for example: 'change email')."
      )
      return "Number of guests set to {}.\n\n{}".format(slots["num_guests"], summary)

  # Step: summary (confirm)
  if step == "summary":
      normalized = user_text.strip().lower()
      if normalized in ("yes", "y", "confirm", "confirmed"):
          ok, res = persist_booking_and_send_confirmation(session_state)
          if ok:
              bs["confirmed"] = True
              bs["booking_id"] = res
              session_state["booking_state"] = bs
              return f"Your booking is confirmed. Booking ID: {res}. A confirmation email has been sent to {slots.get('email')}. Thank you and have a wonderful stay."
          else:
              return f"Unable to save booking: {res}. Your details were NOT saved. Please try again or contact support."
      elif normalized.startswith("change "):
          # User wants to update a single field — store which one and ask for the new value.
          field = normalized.split(" ", 1)[1].strip().lower()

          # Only set pending flag. Do NOT clear slots. Do NOT change step.
          # The NEXT user message will be applied by apply_single_change_and_summary().
          bs["pending_change_field"] = field
          session_state["booking_state"] = bs

          return f"Understood. Please provide the new {field.lower()}."

      elif normalized in ("no", "n", "cancel"):
          session_state["booking_state"] = None
          return "Booking process cancelled as requested. If you would like to start again, say 'I want to book a room'."
      else:
          return "I did not understand. Please reply with **Yes** to confirm and save the booking, reply **No** to cancel, or 'change <field>' to update a field (for example 'change email')."

  return "Unexpected booking step."

# -------------------------
# Intent detection heuristics
# -------------------------
QUESTION_WORDS = ("what", "which", "who", "when", "where", "why", "how", "do", "does", "is", "are", "can")

BOOKING_PATTERNS = re.compile(
  r"\b("
  r"(i\s+want\s+to\s+book)|"
  r"(i'd\s+like\s+to\s+book)|"
  r"(i\s+want\s+to\s+reserve)|"
  r"(make\s+a\s+reservation)|"
  r"(can\s+i\s+book)|"
  r"(may\s+i\s+book)|"
  r"(reserve\s+a)|"
  r"(book\s+a)|"
  r"(book\s+my)|"
  r"(reserve\s+my)|"
  r"(i\s+want\s+to\s+book\s+a)"
  r")\b",
  flags=re.IGNORECASE,
)

def looks_like_info_question(text: str) -> bool:
  if not text or not text.strip():
      return False
  t = text.strip().lower()
  if "?" in text:
      return True
  if any(t.startswith(w + " ") for w in QUESTION_WORDS):
      return True
  if t.startswith(("tell me", "list", "show", "describe", "give me")):
      return True
  if "types of" in t or "room types" in t or "amenities" in t or "what kinds of" in t or "what kinds" in t:
      return True
  return False

def looks_like_booking_intent(text: str) -> bool:
  if not text or not text.strip():
      return False
  t = text.strip().lower()
  if looks_like_info_question(t):
      return False
  if BOOKING_PATTERNS.search(t):
      return True
  # allow "book room" command-like intent if not phrased as a question
  if (("book " in t or "reserve " in t) and ("room" in t or "hotel" in t or "reservation" in t)) \
     and not any(t.startswith(w + " ") for w in QUESTION_WORDS):
      return True
  return False




# -------------------------
# Email sending (lazy import to avoid circular imports)
# returns (ok: bool, msg: str)
# -------------------------
DEV_MODE = os.environ.get("DEV_MODE", "").lower() in ("1", "true", "yes")




def send_verification_email(email: str, code: str) -> Tuple[bool, str]:
  """
  Try to send an email via utils.tools.email_tool (SendGrid wrapper).
  Returns (ok, message). In DEV_MODE the code is shown in UI and function returns (True, msg).
  """
  # If dev/test, show code in UI and avoid external API
  if DEV_MODE:
      try:
          st.info(f"[DEV MODE] Verification code for {email}: {code}")
      except Exception:
          pass
      return True, f"DEV_MODE: code {code}"




  # lazy import to avoid circular import if utils.tools imports other project modules
  try:
      from utils.tools import email_tool
  except Exception as e:
      err = f"ImportError when importing email_tool: {e}"
      print("[EMAIL DEBUG]", err)
      try:
          st.error(err)
      except Exception:
          pass
      return False, err




  subject = "Verification code for your booking"
  body = (
      f"Dear Customer,\n\n"
      f"Your verification code for the booking process is: {code}\n\n"
      f"This code will expire in {int(VERIFICATION_TTL.total_seconds() / 60)} minutes.\n\n"
      f"Regards,\nHotel Reservations Team\n"
  )

  try:
      ok, msg = email_tool(email, subject, body)
      # Ensure tuple-like return
      if ok:
          return True, msg or "sent"
      else:
          return False, msg or "send failed"
  except Exception as e:
      err = f"Exception while calling email_tool: {type(e).__name__}: {e}"
      print("[EMAIL DEBUG]", err)
      try:
          st.error(err)
      except Exception:
          pass
      return False, err


# -------------------------
# Persist booking & confirm (lazy import)
# -------------------------
def persist_booking_and_send_confirmation(session_state: dict) -> Tuple[bool, str]:
  try:
      from utils.tools import booking_persistence_tool, email_tool
  except Exception as e:
      msg = f"Import error: {e}"
      print("[IMPORT DEBUG]", msg)
      return False, msg

  bs = session_state.get("booking_state")
  if not bs:
      return False, "No active booking."

  try:
      ok, booking_id = booking_persistence_tool(bs)
  except Exception as e:
      return False, str(e)

  if not ok:
      return False, booking_id or "failed to persist"

  bs["booking_id"] = booking_id
  bs["confirmed"] = True
  session_state["booking_state"] = bs

  # attempt to send confirmation email (best-effort)
  try:
      subject = f"Booking Confirmation #{booking_id}"
      body = (
          f"Dear {bs['slots'].get('name')},\n\n"
          f"Your booking is confirmed.\n\n"
          f"Booking ID: {booking_id}\n"
          f"Room: {bs['slots'].get('room_type')}\n"
          f"Check-in: {bs['slots'].get('check_in_date')}\n"
          f"Check-out: {bs['slots'].get('check_out_date')}\n\n"
          f"Regards,\nHotel Reservations Team\n"
      )
      # Use email_tool directly for confirmation
      email_ok, _ = email_tool(bs['slots'].get('email'), subject, body)
      # return booking id even if email fails
      return True, booking_id
  except Exception:
      return True, booking_id

# -------------------------
# UI pages
# -------------------------
def admin_page():
    st.title(" Admin Dashboard – Hotel Bookings")

    with st.expander("Filters", expanded=True):

        st.subheader("Select a filter")

        filter_type = st.selectbox(
            "Choose filter category:",
            [
                "None",
                "Guest Name",
                "Email",
                "Check-in Date",
                "Check-out Date",
                "Booking Creation Date",
                "Number of Guests",
                "Room Type",
            ]
        )

        # Initialize variables
        name_filter = email_filter = checkin_filter = checkout_filter = None
        booking_date_filter = num_guests_filter = room_type_filter = None

        if filter_type == "Guest Name":
            name_filter = st.text_input("Enter guest name")

        elif filter_type == "Email":
            email_filter = st.text_input("Enter email")

        elif filter_type == "Check-in Date":
            checkin_filter = st.date_input("Select check-in date")

        elif filter_type == "Check-out Date":
            checkout_filter = st.date_input("Select check-out date")

        elif filter_type == "Booking Creation Date":
            booking_date_filter = st.date_input("Select booking creation date")

        elif filter_type == "Number of Guests":
            num_guests_filter = st.number_input("Enter number of guests", min_value=1, step=1)

        elif filter_type == "Room Type":
            room_type_filter = st.text_input("Enter room type (e.g., Deluxe, Suite)")

        st.subheader("Sorting Options")

        sort_column = st.selectbox(
            "Sort by",
            [
                "name",
                "email",
                "phone",
                "room_type",
                "check_in_date",
                "check_out_date",
                "num_guests",
                "created_at",
                "id",
            ]
        )

        sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
        sort_desc = (sort_order == "Descending")

    # Date → string
    # Convert active filters to strings
    checkin_str = checkin_filter.strftime("%Y-%m-%d") if checkin_filter else None
    checkout_str = checkout_filter.strftime("%Y-%m-%d") if checkout_filter else None
    booking_date_str = booking_date_filter.strftime("%Y-%m-%d") if booking_date_filter else None


    # Query DB
    rows = list_bookings(
        name_filter=name_filter,
        email_filter=email_filter,
        checkin_filter=checkin_str,
        checkout_filter=checkout_str,
        booking_date_filter=booking_date_str,
        num_guests_filter=num_guests_filter,
        room_type_filter=room_type_filter,
        sort_column=sort_column,
        sort_desc=sort_desc     
    )

    st.write(f"Found **{len(rows)}** matching bookings.")
    df = None

    if rows:
        # Convert rows to DataFrame
        import pandas as pd
        df = pd.DataFrame([dict(r) for r in rows])
        st.dataframe(df.set_index("id"), use_container_width=True)

        # Export CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export Results to CSV",
            data=csv,
            file_name="booking_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No matching bookings found.")

# -------------------------
# Booking session init
# -------------------------
def init_booking_state(session_state: dict):
  session_state["booking_state"] = {
      "active": True,
      "step": "ask_name",
      "slots": {
          "name": None,
          "email": None,
          "phone": None,
          "room_type": None,
          "check_in_date": None,
          "check_out_date": None,
          "num_guests": None,
      },
      "verification_code": None,
      "verification_sent_at": None,
      "verified": False,
      "confirmed": False,
      "booking_id": None
  }

def chat_page():
   st.title("AI Hotel Booking Assistant")
   st.markdown("**Welcome guest! How may I assist you today?** Upload the hotel brochure to enable the chat feature.")


   st_html("""
   <script>
   const scrollToBottom = () => {
       window.scrollTo(0, document.body.scrollHeight);
   }


   setInterval(scrollToBottom, 200);   // keeps forcing scroll down
   </script>
   """, height=0)


   # ---- File uploader ----
   uploaded = st.file_uploader(
       "Upload one or more PDFs (small files recommended)",
       type=["pdf"],
       accept_multiple_files=True,
   )

   # ---- Session init ----
   if "tfidf_cache" not in st.session_state:
       st.session_state.tfidf_cache = {}
   if "vector_store" not in st.session_state:
       st.session_state.vector_store = None
   if "messages" not in st.session_state:
       st.session_state.messages = []
   if "booking_state" not in st.session_state:
       st.session_state.booking_state = None


   # ---- Handle PDF uploads ----
   if uploaded:
       files_bytes = [f.read() for f in uploaded]
       key = "_".join([file_sha256(b)[:8] for b in files_bytes])


       if key in st.session_state.tfidf_cache:
        store = st.session_state.tfidf_cache[key]
        hotel_name = store.get("hotel_name", "the hotel")
    
        # ensure room options are restored from cached store
        st.session_state["room_options"] = store.get("room_options", []) or []

        st.session_state.vector_store = store
        st.success(f"Welcome to {hotel_name}. How may I assist you?")
        
       else:
        progress = st.progress(0)
        status = st.empty()
        status.info("Reading documents...")


        try:
               combined = b"\n".join(files_bytes)
               # Primary extractor
               raw = extract_text_from_pdf_bytes(combined)
               text = "\n".join(raw) if isinstance(raw, list) else str(raw or "")
              
               # If cloud returned garbage → fallback to pdfplumber
               if (not text) or (len(text.strip()) < 150) or ("Guests" in text[:50]):
                  from models.embeddings import safe_extract_pdf_text
                  text = safe_extract_pdf_text(combined)

               progress.progress(20)


               detected_hotel = extract_hotel_name_strict(text)

               try:
                   room_opts = extract_room_types_strict(text)
               except:
                   room_opts = []

               st.session_state["room_options"] = room_opts or []


               if not text.strip():
                   progress.empty()
                   status.warning("No readable text found.")
                   st.session_state.vector_store = None
               else:
                   def cb(done, total):
                       pct = int(20 + (done/total)*80)
                       progress.progress(pct)


                   store = build_local_tfidf_store_from_text(text, 1200, 200, cb)

                   raw_filename_title = clean_filename_title(uploaded[0].name)
                   hotel_name = detected_hotel or raw_filename_title
                   store["hotel_name"] = hotel_name
                   store["room_options"] = st.session_state["room_options"]

                   st.session_state.tfidf_cache[key] = store
                   st.session_state.vector_store = store

                   progress.empty()
                   status.success(f"Welcome to {hotel_name}. How may I help you?")
                   scroll_to_bottom()


        except:
               progress.empty()
               status.error("Error reading PDF.")

   # ---- One-time quit instruction ----
   if len(st.session_state.messages) == 0:
    notice_msg = (
        "*Tip:* You can type **end**, **quit**, or **exit** anytime to end the current session.\n"
        "You can always start a new conversation afterward."
    )

    # Display only, DO NOT add to messages (prevents duplication)
    with st.chat_message("assistant"):
        st.markdown(notice_msg)

    # Mark as shown so it won’t appear again
    st.session_state.shown_exit_notice = True



   # ---- Show chat history ----
   for m in st.session_state.messages:
       with st.chat_message(m["role"]):
           st.markdown(m["content"])


   scroll_to_bottom()

   # ============================================================
   #                       DATE PICKER MODE 
   # ============================================================
   bs = st.session_state.get("booking_state")
   if bs and bs.get("active"):
       step = bs["step"]
       slots = bs["slots"]

       # ------------------------------------------------------------
       #                    CHECK-IN CALENDAR
       # ------------------------------------------------------------
       if step == "ask_checkin":
           st.markdown("**Select your check-in date:**")


           checkin = st.date_input(
               "Choose date",
               key="chk_in_picker",
               min_value=datetime.utcnow().date()
           )

           scroll_to_bottom()

           if st.button("Confirm Check-in Date"):
               date_text = checkin.strftime("%Y-%m-%d")


               # show user message
               st.session_state.messages.append({"role": "user", "content": date_text})
               with st.chat_message("user"):
                   st.markdown(date_text)


               reply = handle_booking_step(date_text, st.session_state)
               safe_append_assistant_reply(reply)
               with st.chat_message("assistant"):
                   st.markdown(reply)


               scroll_to_bottom()
               st.rerun()

           return  # Disable chat_input entirely
       # ------------------------------------------------------------
       #        CHECK-OUT CALENDAR  (WORKING MODEL)
       # ------------------------------------------------------------
       if step == "ask_checkout":
           st.markdown("**Select your check-out date:**")


           min_checkout = datetime.utcnow().date()
           if slots.get("check_in_date"):
               try:
                   min_checkout = datetime.strptime(slots["check_in_date"], "%Y-%m-%d").date()
               except:
                   pass


           checkout = st.date_input(
               "Choose date",
               key="chk_out_picker",
               min_value=min_checkout
           )

           scroll_to_bottom()

           if st.button("Confirm Check-out Date"):
               date_text = checkout.strftime("%Y-%m-%d")

               # show user message
               st.session_state.messages.append({"role": "user", "content": date_text})
               with st.chat_message("user"):
                   st.markdown(date_text)

               reply = handle_booking_step(date_text, st.session_state)
               safe_append_assistant_reply(reply)
               with st.chat_message("assistant"):
                   st.markdown(reply)

               scroll_to_bottom()
               st.rerun()

           return  # Disable chat_input entirely

   # ============================================================
   # Chat input (ONLY when not in calendar mode)
   # ============================================================
   if not st.session_state.get("vector_store"):
    st.chat_input(
        "Please upload a hotel brochure to start chatting…",
        disabled=True
    )
    return

  # If brochure is uploaded → enable chat
   user_input = st.chat_input("Ask something or say 'I want to book a room'…")
   if not user_input:
      return

   user_text = user_input.strip()

   # ---- End / Quit session detection ----
   if re.fullmatch(r"(end|quit|exit|stop|bye|end session)", user_text, flags=re.IGNORECASE):
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Reset booking flow state
        st.session_state.booking_state = None

        goodbye = (
            "Thank you for chatting with me. \n\n"
            "Your session has now ended successfully.\n\n"
            "**To start a NEW conversation**, simply type:\n"
            "- *hello*\n"
            "- *hi*\n"
            "- *i want to book a room*\n"
            "- *start*\n\n"
            "I’ll be right here whenever you need assistance again. "
        )

        st.session_state.messages.append({"role": "assistant", "content": goodbye})
        with st.chat_message("assistant"):
            st.markdown(goodbye)

        scroll_to_bottom()
        return

    # ---- Start new conversation on greeting ----
   if re.fullmatch(r"(hello|hi|hey|start|restart|begin)", user_text, flags=re.IGNORECASE):
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # reset booking flow
        st.session_state.booking_state = None

        # friendly restart message
        reply = (
            "Hello again! 👋\n\n"
            "I'm ready to assist you.\n"
            "You may ask a question about the hotel brochure, or say **I want to book a room** to start a new booking."
        )

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

        scroll_to_bottom()
        return


   # ---- Thank-you detection ----
   if re.search(r"\b(thank\s*you|thanks|thank u|thx|ty)\b", user_text, re.IGNORECASE):
       st.session_state.messages.append({"role": "user", "content": user_text})
       with st.chat_message("user"):
           st.markdown(user_text)

       reply = "I am happy to be of assistance. What else can I do for you?"
       with st.chat_message("assistant"):
           st.markdown(reply)

       safe_append_assistant_reply(reply)
       scroll_to_bottom()
       return

   # ---- Info question detection ----
   INFO_RE = re.compile(r"\b(what|which|when|where|why|how|tell|show|describe|list|types|rooms|amenities)\b", re.IGNORECASE)
   looks_info = "?" in user_text or INFO_RE.search(user_text)


   if looks_info:
       st.session_state.messages.append({"role": "user", "content": user_text})
       with st.chat_message("user"):
           st.markdown(user_text)


       reply, st.session_state = handle_user_message(user_text, st.session_state)
       with st.chat_message("assistant"):
           st.markdown(reply)


       safe_append_assistant_reply(reply)
       scroll_to_bottom()
       return

   # ---- Booking logic ----
   st.session_state.messages.append({"role": "user", "content": user_text})
   with st.chat_message("user"):
       st.markdown(user_text)

   bs = st.session_state.get("booking_state")
   looks_booking = looks_like_booking_intent(user_text)

   if (not bs or not bs["active"]) and looks_booking:
       init_booking_state(st.session_state)
       msg = "Certainly. Please provide your full name."
       safe_append_assistant_reply(msg)
       with st.chat_message("assistant"):
           st.markdown(msg)
       scroll_to_bottom()
       return

   if bs and bs["active"]:
       reply = handle_booking_step(user_text, st.session_state)
       safe_append_assistant_reply(reply)
       with st.chat_message("assistant"):
           st.markdown(reply)
       scroll_to_bottom()
       return

   # ---- Default RAG fallback ----
   reply, st.session_state = handle_user_message(user_text, st.session_state)
   with st.chat_message("assistant"):
       st.markdown(reply)

   safe_append_assistant_reply(reply)
   scroll_to_bottom()

st_html("<script>window.scrollTo(0, document.body.scrollHeight);</script>", height=0)

# -------------------------
# App init & main
# -------------------------
def init_app_state():
  if "messages" not in st.session_state:
      st.session_state.messages = []
  if "booking_state" not in st.session_state:
      st.session_state.booking_state = None
  if "vector_store" not in st.session_state:
      st.session_state.vector_store = None
  if "tfidf_cache" not in st.session_state:
      st.session_state.tfidf_cache = {}
  if "shown_exit_notice" not in st.session_state:
      st.session_state.shown_exit_notice = False

     # always ensure the room options key exists (avoid attribute/dict ambiguity)
  if "room_options" not in st.session_state:
      st.session_state["room_options"] = []

def main():
  st.set_page_config(page_title="Hotel Booking Assistant", page_icon="🏨", layout="wide")
  init_app_state()
  init_db()

  with st.sidebar:
      st.title("Navigation")
      page = st.radio("Go to:", ["Chat", "Admin Dashboard"])
      st.markdown("---")
      if st.button("Clear Chat"):
          st.session_state.messages = []
          st.session_state.booking_state = None
          st.experimental_rerun()
      st.markdown("### Status")
      if st.session_state.get("vector_store"):
          st.success("Document store ready")
      else:
          st.info("No document store loaded")

  if page == "Chat":
      chat_page()
  elif page == "Admin Dashboard":
    admin_page()

if __name__ == "__main__":
  main()





