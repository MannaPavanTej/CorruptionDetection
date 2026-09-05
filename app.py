"""
app.py
------

Flask application for an AI-based Corruption Detection System.

Main responsibilities:
- Provide authentication (admin and officer users)
- Manage government schemes and transactions
- Call the trained RandomForest model to predict corruption probability
- Store corruption reports
- Render a dashboard with summary stats and a Chart.js bar chart
"""

import os
import pickle
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "corruption_model.pkl")


app = Flask(__name__)

# NOTE: Use a more secure secret key for production.
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-secret-key-change-in-production"
)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DATABASE_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'admin' or 'officer'

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Scheme(db.Model):
    __tablename__ = "schemes"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False, unique=True)
    allocated_amount = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=True)

    transactions = db.relationship("Transaction", backref="scheme", lazy=True)


class Transaction(db.Model):
    __tablename__ = "transactions"

    id = db.Column(db.Integer, primary_key=True)
    scheme_id = db.Column(db.Integer, db.ForeignKey("schemes.id"), nullable=False)
    beneficiary_name = db.Column(db.String(120), nullable=False)
    expected_amount = db.Column(db.Float, nullable=False)
    transferred_amount = db.Column(db.Float, nullable=False)
    amount_difference = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(20), nullable=False)  # store as ISO date string
    delay_days = db.Column(db.Integer, nullable=False)

    corruption_probability = db.Column(db.Float, nullable=True)
    is_corruption_detected = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class CorruptionReport(db.Model):
    __tablename__ = "corruption_reports"

    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey("transactions.id"))
    scheme_id = db.Column(db.Integer, db.ForeignKey("schemes.id"))

    location = db.Column(db.String(120), nullable=False)
    corruption_probability = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Machine Learning Model Loading & Prediction
# ---------------------------------------------------------------------------
class CorruptionModel:
    """
    Wrapper for the trained RandomForest model and location encoder.
    Handles feature preparation and prediction.
    """

    def __init__(self, model_path: str):
        self.model = None
        self.location_encoder = None
        self.feature_columns = None
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            app.logger.warning(
                "ML model file not found at %s. "
                "Make sure to run model_training.py before starting the app.",
                model_path,
            )
            return
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model = bundle.get("model")
        self.location_encoder = bundle.get("location_encoder")
        self.feature_columns = bundle.get("feature_columns")
        app.logger.info("ML model loaded successfully from %s", model_path)

    def is_ready(self) -> bool:
        return self.model is not None and self.location_encoder is not None

    def predict_proba(
        self,
        scheme_id: int,
        expected_amount: float,
        transferred_amount: float,
        delay_days: int,
        location: str,
    ) -> float:
        """
        Predict the probability that a transaction is corrupt.

        Returns a float in [0, 1]. If the model is not ready, returns 0.0.
        """
        if not self.is_ready():
            app.logger.warning("ML model is not available; returning probability 0.0")
            return 0.0

        amount_difference = transferred_amount - expected_amount

        # Map unknown locations to 'Other' (which is included in training).
        classes = list(self.location_encoder.classes_)
        if "Other" not in classes:
            classes.append("Other")
        if location not in self.location_encoder.classes_:
            safe_location = "Other"
        else:
            safe_location = location

        # LabelEncoder cannot dynamically add new classes. We handle the "Other"
        # case by mapping unseen to "Other" if it was part of training.
        loc_encoded = self.location_encoder.transform([safe_location])[0]

        features = [
            float(scheme_id),
            float(expected_amount),
            float(transferred_amount),
            float(amount_difference),
            float(delay_days),
            float(loc_encoded),
        ]

        proba = self.model.predict_proba([features])[0][1]
        return float(proba)


corruption_model = CorruptionModel(MODEL_PATH)


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------
@app.context_processor
def inject_globals():
    """
    Make common variables available in all templates:
    - user: currently logged-in user
    - current_year: year for footer
    """
    return {
        "user": current_user(),
        "current_year": datetime.utcnow().year,
    }


def current_user():
    """Return the currently logged-in user (or None)."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    return User.query.get(user_id)


def login_required(view_func):
    """Decorator to enforce login on routes."""

    from functools import wraps

    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not current_user():
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped


def role_required(*roles):
    """
    Decorator to enforce that the current user has one of the given roles.
    Example: @role_required('admin')
    """

    from functools import wraps

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            user = current_user()
            if not user or user.role not in roles:
                flash("You are not authorized to access this resource.", "danger")
                return redirect(url_for("dashboard"))
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


# ---------------------------------------------------------------------------
# Routes: Authentication
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Redirect root to login or dashboard."""
    if current_user():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Admin/officer login."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            session["role"] = user.role
            flash(f"Welcome, {user.username}!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """User registration - creates new user account."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        role = request.form.get("role", "officer").strip()

        # Validation
        if not username or not password:
            flash("Username and password are required.", "warning")
        elif len(username) < 3:
            flash("Username must be at least 3 characters long.", "warning")
        elif len(password) < 6:
            flash("Password must be at least 6 characters long.", "warning")
        elif password != confirm_password:
            flash("Passwords do not match.", "warning")
        elif User.query.filter_by(username=username).first():
            flash("Username already exists. Please choose a different one.", "warning")
        elif role not in ["admin", "officer"]:
            flash("Invalid role selected.", "warning")
        else:
            # Create new user
            new_user = User(username=username, role=role)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash(f"Account created successfully! You can now login as {username}.", "success")
            return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/logout")
def logout():
    """Log out the current user."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Routes: Dashboard
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    """Show system overview and Chart.js corruption-by-location."""
    total_schemes = Scheme.query.count()
    total_transactions = Transaction.query.count()
    corruption_cases = Transaction.query.filter_by(is_corruption_detected=True).count()

    # Corruption by location
    location_stats = (
        db.session.query(Transaction.location, db.func.count(Transaction.id))
        .filter(Transaction.is_corruption_detected == True)  # noqa: E712
        .group_by(Transaction.location)
        .all()
    )

    locations = [row[0] for row in location_stats]
    counts = [row[1] for row in location_stats]

    latest_transactions = (
        Transaction.query.order_by(Transaction.created_at.desc()).limit(5).all()
    )

    return render_template(
        "dashboard.html",
        user=current_user(),
        total_schemes=total_schemes,
        total_transactions=total_transactions,
        corruption_cases=corruption_cases,
        chart_locations=locations,
        chart_counts=counts,
        latest_transactions=latest_transactions,
    )


# ---------------------------------------------------------------------------
# Routes: Scheme Management
# ---------------------------------------------------------------------------
@app.route("/schemes", methods=["GET", "POST"])
@login_required
@role_required("admin")
def schemes():
    """
    Admin-only scheme management.
    GET: list schemes
    POST: create a new scheme
    """
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        allocated_amount = request.form.get("allocated_amount", "0").strip()
        description = request.form.get("description", "").strip()

        if not name:
            flash("Scheme name is required.", "warning")
        else:
            try:
                allocated_amount_value = float(allocated_amount)
            except ValueError:
                flash("Allocated amount must be a number.", "warning")
            else:
                scheme = Scheme(
                    name=name,
                    allocated_amount=allocated_amount_value,
                    description=description or None,
                )
                db.session.add(scheme)
                db.session.commit()
                flash("Scheme created successfully.", "success")
                return redirect(url_for("schemes"))

    schemes_list = Scheme.query.order_by(Scheme.id.desc()).all()
    return render_template(
        "schemes.html",
        user=current_user(),
        schemes=schemes_list,
    )


# ---------------------------------------------------------------------------
# Routes: Transaction Management
# ---------------------------------------------------------------------------
@app.route("/transactions")
@login_required
def transactions():
    """List all transactions."""
    txs = Transaction.query.order_by(Transaction.created_at.desc()).all()
    return render_template(
        "transactions.html",
        user=current_user(),
        transactions=txs,
    )


@app.route("/transactions/add", methods=["GET", "POST"])
@login_required
def add_transaction():
    """
    Add a new transaction.

    On successful POST, the system:
    - Saves the transaction
    - Calls the ML model for corruption probability
    - Updates the transaction with the probability and corruption flag
    - Creates a record in corruption_reports if probability > threshold
    """
    schemes_list = Scheme.query.order_by(Scheme.name.asc()).all()

    if not schemes_list:
        flash("Please create at least one scheme before adding transactions.", "warning")
        return redirect(url_for("schemes"))

    if request.method == "POST":
        scheme_id = request.form.get("scheme_id")
        beneficiary_name = request.form.get("beneficiary_name", "").strip()
        expected_amount = request.form.get("expected_amount", "0").strip()
        transferred_amount = request.form.get("transferred_amount", "0").strip()
        location = request.form.get("location", "").strip() or "Other"
        date = request.form.get("date", "").strip()
        delay_days = request.form.get("delay_days", "0").strip()

        try:
            scheme_id_int = int(scheme_id)
            expected_amount_val = float(expected_amount)
            transferred_amount_val = float(transferred_amount)
            delay_days_int = int(delay_days)
        except (TypeError, ValueError):
            flash("Please provide valid numeric values.", "warning")
            return render_template(
                "add_transaction.html",
                user=current_user(),
                schemes=schemes_list,
            )

        amount_difference = transferred_amount_val - expected_amount_val
        tx_date = date or datetime.utcnow().strftime("%Y-%m-%d")

        transaction = Transaction(
            scheme_id=scheme_id_int,
            beneficiary_name=beneficiary_name,
            expected_amount=expected_amount_val,
            transferred_amount=transferred_amount_val,
            amount_difference=amount_difference,
            location=location,
            date=tx_date,
            delay_days=delay_days_int,
        )
        db.session.add(transaction)
        db.session.commit()  # commit to get an ID

        # Predict corruption probability using the ML model
        probability = corruption_model.predict_proba(
            scheme_id=scheme_id_int,
            expected_amount=expected_amount_val,
            transferred_amount=transferred_amount_val,
            delay_days=delay_days_int,
            location=location,
        )

        threshold = 0.6
        is_corrupt = probability >= threshold

        transaction.corruption_probability = probability
        transaction.is_corruption_detected = is_corrupt
        db.session.commit()

        if is_corrupt:
            report = CorruptionReport(
                transaction_id=transaction.id,
                scheme_id=transaction.scheme_id,
                location=transaction.location,
                corruption_probability=probability,
            )
            db.session.add(report)
            db.session.commit()
            flash(
                f"Transaction saved. Corruption detected with probability "
                f"{probability:.2%}.",
                "danger",
            )
        else:
            flash(
                f"Transaction saved. Corruption probability: {probability:.2%}.",
                "success",
            )

        return redirect(url_for("transactions"))

    return render_template(
        "add_transaction.html",
        user=current_user(),
        schemes=schemes_list,
    )


# ---------------------------------------------------------------------------
# Routes: Corruption Reports
# ---------------------------------------------------------------------------
@app.route("/reports")
@login_required
def reports():
    """List corruption reports."""
    reports_list = (
        db.session.query(CorruptionReport, Transaction, Scheme)
        .join(Transaction, CorruptionReport.transaction_id == Transaction.id)
        .join(Scheme, CorruptionReport.scheme_id == Scheme.id)
        .order_by(CorruptionReport.created_at.desc())
        .all()
    )
    return render_template(
        "corruption_reports.html",
        user=current_user(),
        reports=reports_list,
    )


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------
def create_default_admin():
    """
    Create a default admin user if no users exist.

    Admin credentials are read from environment variables.
    """

    if User.query.count() == 0:
        admin_username = os.environ.get("ADMIN_USERNAME", "admin")
        admin_password = os.environ.get("ADMIN_PASSWORD")

        if not admin_password:
            app.logger.warning(
                "ADMIN_PASSWORD is not set. Default admin account was not created."
            )
            return

        admin = User(
            username=admin_username,
            role="admin"
        )
        admin.set_password(admin_password)

        db.session.add(admin)
        db.session.commit()

        app.logger.info(
            "Default admin user created (username='%s').",
            admin_username
        )


def init_db():
    """Create database tables and ensure a default admin user."""
    db.create_all()
    create_default_admin()


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        init_db()

    # Enable debug for development; disable in production.
    app.run(debug=True)

