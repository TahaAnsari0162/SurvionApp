# Suspicious Activity Detection Tool

A Flask-based web application for detecting and monitoring suspicious activities in your system. The application features a modern, responsive UI with real-time activity monitoring and user authentication.

## Features

- User authentication (login/register)
- Animated intro page showing suspicious activities
- Real-time activity monitoring dashboard
- Activity statistics and risk assessment
- System status monitoring
- Modern, responsive UI design

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd suspicious-activity-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your virtual environment is activated (if you created one)

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
suspicious-activity-detection/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/         # HTML templates
│   ├── intro.html     # Animated intro page
│   ├── login.html     # Login page
│   ├── register.html  # Registration page
│   └── dashboard.html # Main dashboard
└── README.md         # This file
```

## Security Features

- Password hashing using Werkzeug
- Session management with Flask-Login
- SQLite database for user storage
- Protected routes requiring authentication

## Contributing

Feel free to submit issues and enhancement requests! 