# utils/database.py
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from config.config import settings

def get_connection() -> sqlite3.Connection:
    db_path = Path(settings.DB_PATH)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS guests (
            guest_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_id INTEGER NOT NULL,
            room_type TEXT NOT NULL,
            check_in_date TEXT NOT NULL,
            check_out_date TEXT NOT NULL,
            num_guests INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (guest_id) REFERENCES guests(guest_id)
        );
        """
    )

    conn.commit()
    conn.close()


def get_or_create_guest(name: str, email: str, phone: str) -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT guest_id FROM guests WHERE email = ?", (email.strip(),))
    row = cur.fetchone()
    if row:
        guest_id = row["guest_id"]
    else:
        cur.execute(
            "INSERT INTO guests (name, email, phone) VALUES (?, ?, ?)",
            (name.strip(), email.strip(), phone.strip()),
        )
        conn.commit()
        guest_id = cur.lastrowid

    conn.close()
    return guest_id


def create_booking(
    guest_id: int,
    room_type: str,
    check_in_date: str,
    check_out_date: str,
    num_guests: int,
    status: str = "confirmed",
) -> int:
    conn = get_connection()
    cur = conn.cursor()

    created_at = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO bookings (guest_id, room_type, check_in_date,
                              check_out_date, num_guests, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (guest_id, room_type, check_in_date, check_out_date, num_guests, status, created_at),
    )
    conn.commit()
    booking_id = cur.lastrowid
    conn.close()
    return booking_id


def list_bookings(
    name_filter=None,
    email_filter=None,
    checkin_filter=None,
    checkout_filter=None,
    booking_date_filter=None,
    num_guests_filter=None,
    room_type_filter=None,
    sort_column="created_at",
    sort_desc=True,
):
    conn = get_connection()
    cur = conn.cursor()

    # Base JOIN: bookings + guests
    query = """
        SELECT 
            b.id,
            g.guest_id,
            g.name,
            g.email,
            g.phone,
            b.room_type,
            b.check_in_date,
            b.check_out_date,
            b.num_guests,
            b.created_at
        FROM bookings b
        JOIN guests g ON b.guest_id = g.guest_id
        WHERE 1=1
    """

    params = []

    # Filters
    if name_filter:
        query += " AND g.name LIKE ?"
        params.append(f"%{name_filter}%")

    if email_filter:
        query += " AND g.email LIKE ?"
        params.append(f"%{email_filter}%")

    if checkin_filter:
        query += " AND b.check_in_date = ?"
        params.append(checkin_filter)

    if checkout_filter:
        query += " AND b.check_out_date = ?"
        params.append(checkout_filter)

    if booking_date_filter:
        query += " AND b.created_at LIKE ?"
        params.append(f"{booking_date_filter}%")

    if num_guests_filter:
        query += " AND b.num_guests = ?"
        params.append(num_guests_filter)

    if room_type_filter:
        query += " AND b.room_type LIKE ?"
        params.append(f"%{room_type_filter}%")

    # Sorting logic
    sortable_columns = {
        "name": "g.name",
        "email": "g.email",
        "phone": "g.phone",
        "room_type": "b.room_type",
        "check_in_date": "b.check_in_date",
        "check_out_date": "b.check_out_date",
        "num_guests": "b.num_guests",
        "created_at": "b.created_at",
        "id": "b.id",
    }

    sort_col = sortable_columns.get(sort_column, "b.created_at")
    direction = "DESC" if sort_desc else "ASC"

    query += f" ORDER BY {sort_col} {direction}"

    cur.execute(query, params)
    rows = cur.fetchall()

    conn.close()
    return rows


