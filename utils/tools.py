# utils/tools.py
import os
import requests
from typing import Tuple

def email_tool(to_address: str, subject: str, body: str) -> Tuple[bool, str]:
    """
    Simple SendGrid send wrapper.
    Returns: (ok: bool, message: str)
    """
    SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
    FROM_EMAIL = os.environ.get("EMAIL_FROM")

    if not SENDGRID_API_KEY:
        return False, "SENDGRID_API_KEY not set"
    if not FROM_EMAIL:
        return False, "EMAIL_FROM not set"

    url = "https://api.sendgrid.com/v3/mail/send"
    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "personalizations": [{"to": [{"email": to_address}], "subject": subject}],
        "from": {"email": FROM_EMAIL},
        "content": [{"type": "text/plain", "value": body}]
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        if 200 <= resp.status_code < 300:
            return True, "Email sent successfully via SendGrid"
        else:
            return False, f"SendGrid error {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, f"Exception when sending email: {e}"

def booking_persistence_tool(booking_state):
    # keep your DB code here; return (True, booking_id) on success or (False, "error")
    from utils.database import get_or_create_guest, create_booking
    try:
        s = booking_state["slots"]
        guest_id = get_or_create_guest(name=s["name"], email=s["email"], phone=s["phone"])
        booking_id = create_booking(
            guest_id=guest_id,
            room_type=s.get("room_type", "N/A"),
            check_in_date=s.get("check_in_date", "N/A"),
            check_out_date=s.get("check_out_date", "N/A"),
            num_guests=int(s.get("num_guests", 1)),
            status="confirmed",
        )
        return True, str(booking_id)
    except Exception as e:
        return False, str(e)
