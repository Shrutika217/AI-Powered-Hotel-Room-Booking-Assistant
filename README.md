# AI Hotel Booking Assistant

The AI Hotel Booking Assistant allows hotels to upload brochure PDFs and instantly provide guests with room information, brochure-based question answering, and a complete booking workflow inside a conversational Streamlit interface.

This README describes the project, architecture, usage, installation, and developer notes.

---

## Overview

The AI Hotel Booking Assistant supports:
- PDF brochure upload
- Strict extraction of hotel name and room types
- Local TF-IDF based question answering (RAG without external APIs)
- Full booking flow with email verification
- Universal quit/end system
- SQLite database for bookings
- Admin dashboard for filtering, sorting, and exporting
- Persistent conversation state using Streamlit session_state

---

## Features

### Brochure Upload

Upload one or more PDF brochures. The system extracts:
- Clean text from the PDF
- Hotel name using strict rules
- Room types only from specific sections such as:
  - “Room Types”
  - “Rooms Available”
  - “Rooms and Suites”
  - “Room Categories”

### Retrieval-Augmented Question Answering (RAG)

Uses a local TF-IDF vector store:
- No external LLM required
- Deterministic, fast responses
- Supports questions like:
  - What amenities are available?
  - What types of rooms are offered?
  - Describe the Deluxe Room.

### Booking Flow

A structured guided booking process:

1. Full name
2. Email address
3. 6-digit Email OTP verification
4. Phone number
5. Room type selection
6. Check-in date selection (calendar widget)
7. Check-out date selection (calendar widget)
8. Number of guests
9. Summary and confirmation
10. Booking saved to SQLite and confirmation email sent

Also supports modification commands to change or ammend any field using the command "change field_name".

### Universal Quit System

Typing any of the following ends the session immediately:
- end
- quit
- exit
- stop
- bye
- end session

The assistant clears booking state and shows instructions for starting a new conversation.

### Admin Dashboard

Allows staff to:
- Filter bookings by guest name, email, check-in date, check-out date, number of guests, room type, and booking creation date.
- Sort results by any field.
- Export filtered results to CSV.
- View results in a live table.

### SQLite Storage

Data is stored in two tables:
1. guests (guest_id, name, email, phone)
2. bookings (id, guest_id, room_type, check_in_date, check_out_date, num_guests, created_at)

## Project Structure

```text
HOTELBOT/
├── .cache_fast_store/
├── .cache_local_store/
├── .pdf_index_cache/
├── .temp_uploads/
│
├── config/
│   ├── __pycache__/
│   └── config.py
│
├── models/
│   ├── __pycache__/
│   ├── embeddings.py
│   ├── llm.py
│   └── pdf_utils.py
│
├── utils/
│   ├── __pycache__/
│   ├── booking_flow.py
│   ├── chat_logic.py
│   ├── database.py
│   └── tools.py
│
├── app.py
├── hotel_bookings.db
└── requirements.txt
```

---

## Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

git clone <[this-repository-url](https://github.com/Shrutika217/AI-Powered-Hotel-Room-Booking-Assistant)>

### 2. Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate               # MacOS / Linux
venv\Scripts\activate                  # Windows

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Environment Variables (Optional)

Create a `.env` file if you need email verification or API keys:

touch .env

Add values:

```env
GEMINI_API_KEY = "your_key_here"
SENDGRID_API_KEY = "your_key_here"
EMAIL_FROM = "hotelbot@example.com"
SMTP_HOST="smtp.sendgrid.net"
SMTP_PORT="587" 
DEV_MODE=0      # for calling the SendGrid API. In developement stage, use DEV_MODE=1 to get the verfication code on the frontend without having to call the API.
```

### 5. Initialize the Database

The database is automatically created when running the app.  
To manually initialize it:

python -c "import utils.database as d; d.init_db()"

### 6. Run the Application

streamlit run app.py

The app will open at:
http://localhost:8501

---

## PDF Upload Instructions

1. Navigate to the Chat page.
2. Upload a hotel brochure PDF.
3. Once processed, the assistant becomes active.
4. You may:
   - Ask questions about the brochure.
   - Begin a booking by typing:
     ```
     I want to book a room
     ```

---

---

## Contributing

Contributions are welcome.  
If you have ideas for improving the extraction logic, booking flow, or admin tools, feel free to open an Issue or submit a Pull Request.

---

## Future Enhancements

Potential areas to expand:

- Pricing and availability extraction  
- Image-based room recognition  
- Multiple PDF brochures per hotel profile  
- Cloud database (PostgreSQL) integration  
- Automated email templates with branding  

---

## License

Distributed under the MIT License.

---

## Contact

For questions or suggestions:

GitHub: https://github.com/Shrutika217  
LinkedIn: https://www.linkedin.com/in/shrutika-gupta-687b55243/

---

Thank you for using the AI Hotel Booking Assistant.






