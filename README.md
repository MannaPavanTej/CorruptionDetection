# Corruption Detection System (Flask + Machine Learning)

An AI-based prototype for identifying potentially suspicious transactions in government schemes using **Flask, SQLite, SQLAlchemy, and a Random Forest machine learning model**.

The system allows administrators and officers to manage government schemes and transactions, while the ML model calculates a corruption probability for each transaction.

> **Note:** This project is a prototype. The current ML model is trained and evaluated using synthetically generated transaction data. Its evaluation results should not be interpreted as real-world corruption-detection accuracy.

---

## Tech Stack

- **Backend:** Flask
- **Database:** SQLite
- **ORM:** Flask-SQLAlchemy
- **Machine Learning:** scikit-learn
- **ML Algorithm:** Random Forest Classifier
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Chart.js
- **Frontend:** HTML, CSS, Bootstrap
- **Authentication:** Flask sessions + Werkzeug password hashing

---

## Features

### Authentication

- Admin login and logout
- Officer login
- Password hashing using Werkzeug
- Role-based access control

### Government Scheme Management

- Add government schemes
- Define allocated scheme amounts
- View available schemes

### Transaction Management

- Add transaction details
- Select government scheme
- Record beneficiary information
- Compare expected and transferred amounts
- Record transaction location
- Record transaction date
- Record payment delay
- Automatically calculate corruption probability using the ML model

### Corruption Detection

The Random Forest model evaluates transaction features and generates a probability indicating whether a transaction may be suspicious.

The system flags transactions when their predicted corruption probability crosses the configured detection threshold.

### Dashboard

The dashboard provides an overview of:

- Total schemes
- Total transactions
- Detected corruption cases
- Corruption distribution by location
- Transaction risk/corruption probability
- Visual charts using Chart.js

### ML Model Evaluation

The training pipeline generates:

- Accuracy
- ROC-AUC score
- Classification report
- Confusion matrix
- ROC curve
- Feature importance chart

---

## Machine Learning

The current model uses a **Random Forest Classifier**.

### Input Features

The model uses transaction-related features including:

- Scheme ID
- Expected amount
- Transferred amount
- Amount difference
- Delay in days
- Transaction location

### Current Model Results

The model was evaluated on a synthetic test dataset containing **400 transactions**.

| Metric | Result |
|---|---:|
| Test Accuracy | 93.75% |
| ROC-AUC | 0.9008 |
| Suspicious Transaction Recall | 80% |

### Evaluation Visualizations

The training script generates the following files:

```text
evaluation/
├── confusion_matrix.png
├── roc_curve.png
└── feature_importance.png

## Project Structure

```text
CorruptionDetection/
│
├── app.py
├── model_training.py
├── seed_data.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── corruption_model.pkl
│
├── evaluation/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── templates/
│   └── ...
│
└── static/
    └── ...