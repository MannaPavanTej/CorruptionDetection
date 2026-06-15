# Corruption Detection System (Flask + ML)

This project is an AI-based **Corruption Detection System** built with:

- **Backend**: Flask
- **Database**: SQLite (via SQLAlchemy)
- **Machine Learning**: scikit-learn (RandomForestClassifier)
- **Frontend**: HTML, CSS, Bootstrap, Chart.js

The system monitors government scheme transactions and predicts possible corruption using a trained machine learning model.

## Features

- **Authentication**
  - Admin login/logout
  - Officer login
- **Scheme Management**
  - Add government schemes
  - Define allocated amount
- **Transaction Management**
  - Add transaction details (scheme, beneficiary, expected vs transferred amount, location, date, delay days)
  - ML-based corruption probability prediction on each new transaction
- **Dashboard**
  - Total schemes
  - Total transactions
  - Number of corruption cases
  - Corruption by location (bar chart using Chart.js)
  - Display fraud probability

## Project Structure

```text
CorruptionDetection/
  app.py
  model_training.py
  requirements.txt
  README.md
  database.db              # created at runtime
  /models                  # trained ML model pickle
  /templates               # HTML templates
  /static                  # CSS, JS, assets
```

## Setup Instructions

1. **Create and activate a virtual environment (recommended)**

```bash
python -m venv venv
venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Train the ML model**

This will create the trained RandomForest model and encoder inside the `models` folder.

```bash
python model_training.py
```

4. **Run the Flask app**

```bash
python app.py
```

The app will start on `http://127.0.0.1:5000/` by default.

5. **Default Credentials**

On first run, the app will create an admin user if none exists:

- **Username**: `admin`
- **Password**: `admin123`
- **Role**: `admin`

You can create officer accounts from within the database or extend the app to provide an admin UI for user creation.

## Notes

- The ML model is trained on a **synthetic sample dataset** defined in `model_training.py`. You can replace this with real, cleaned data as needed.
- The app uses **session-based authentication** with password hashing (Werkzeug).
- All critical sections are commented for clarity and easy extension.

