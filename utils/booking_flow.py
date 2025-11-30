# utils/booking_flow.py
from typing import Dict, Tuple
from datetime import datetime
import re

REQUIRED_SLOTS = [
    "name",
    "email",
    "phone",
    "room_type",
    "check_in_date",
    "check_out_date",
    "num_guests",
]


def initialize_booking_state() -> Dict:
    return {
        "slots": {slot: None for slot in REQUIRED_SLOTS},
        "confirmed": False,
        "booking_id": None,
    }


def get_next_missing_slot(booking_state: Dict) -> str | None:
    for slot in REQUIRED_SLOTS:
        if not booking_state["slots"].get(slot):
            return slot
    return None


def slot_prompt(slot: str) -> str:
    prompts = {
        "name": "Please share your full name.",
        "email": "What is your email address?",
        "phone": "What is your phone number?",
        "room_type": "What type of room would you like? (e.g., Single, Double, Suite)",
        "check_in_date": "What is your check-in date? (YYYY-MM-DD)",
        "check_out_date": "What is your check-out date? (YYYY-MM-DD)",
        "num_guests": "How many guests will be staying?",
    }
    return prompts.get(slot, f"Please provide {slot}.")


# --------- basic validators ----------

def is_valid_email(email: str) -> bool:
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return bool(re.match(pattern, email))


def is_valid_phone(phone: str) -> bool:
    digits = re.sub(r"\D", "", phone)
    return 7 <= len(digits) <= 15


def parse_date(date_str: str):
    return datetime.strptime(date_str, "%Y-%m-%d")


def is_valid_date(date_str: str) -> bool:
    try:
        parse_date(date_str)
        return True
    except Exception:
        return False


def validate_and_store(slot: str, value: str, booking_state: Dict) -> str | None:
    value = value.strip()

    if slot == "email":
        if not is_valid_email(value):
            return "That doesn't look like a valid email. Please enter a correct email address."
    elif slot == "phone":
        if not is_valid_phone(value):
            return "That doesn't look like a valid phone number. Please enter digits like +11234567890."
    elif slot in ["check_in_date", "check_out_date"]:
        if not is_valid_date(value):
            return "Please enter the date in YYYY-MM-DD format."

        # Extra check: check-out must be after check-in (if both filled)
        tmp_slots = dict(booking_state["slots"])
        tmp_slots[slot] = value
        ci = tmp_slots.get("check_in_date")
        co = tmp_slots.get("check_out_date")
        if ci and co:
            if parse_date(co) <= parse_date(ci):
                return "Check-out date must be after check-in date. Please enter a valid date."
    elif slot == "num_guests":
        if not value.isdigit() or int(value) <= 0:
            return "Please enter a positive integer for number of guests."

    booking_state["slots"][slot] = value
    return None  # no error


def summarize_booking(booking_state: Dict) -> str:
    s = booking_state["slots"]
    return (
        "Here are your booking details:\n\n"
        f"- Name: {s['name']}\n"
        f"- Email: {s['email']}\n"
        f"- Phone: {s['phone']}\n"
        f"- Room type: {s['room_type']}\n"
        f"- Check-in date: {s['check_in_date']}\n"
        f"- Check-out date: {s['check_out_date']}\n"
        f"- Number of guests: {s['num_guests']}\n\n"
        "Please type **yes** to confirm or **no** to cancel."
    )


def process_booking_turn(user_message: str, booking_state: Dict) -> Tuple[str, bool]:
    """
    One step of the booking flow.
    Returns: (bot_reply, booking_completed_flag)
    """
    # If all slots are filled but not confirmed, we are waiting for confirmation
    if get_next_missing_slot(booking_state) is None and not booking_state["confirmed"]:
        text = user_message.strip().lower()
        if text in ["yes", "y", "confirm"]:
            booking_state["confirmed"] = True
            return "Great! I will save your hotel booking now and send you a confirmation email.", True
        elif text in ["no", "n", "cancel"]:
            booking_state["confirmed"] = False
            return "Okay, I have cancelled this booking. You can start again anytime.", True
        else:
            return "Please type **yes** to confirm the booking or **no** to cancel.", False

    current_slot = get_next_missing_slot(booking_state)

    # First entry into booking flow: ask for the first slot
    if current_slot and booking_state["slots"][current_slot] is None and not any(
        booking_state["slots"].values()
    ):
        return slot_prompt(current_slot), False

    # Treat user_message as answer to current_slot
    error = validate_and_store(current_slot, user_message, booking_state)
    if error:
        return error, False

    # Next slot?
    next_slot = get_next_missing_slot(booking_state)
    if next_slot:
        return slot_prompt(next_slot), False
    else:
        summary = summarize_booking(booking_state)
        return summary, False
