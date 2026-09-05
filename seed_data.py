"""
seed_data.py
------------
Seeds the database with dummy government schemes and transactions,
then runs the ML model on each transaction to set corruption probability
and "Corruption Detected" flags. Run after model_training.py.

Usage:
  python seed_data.py          # add data only if DB is empty
  python seed_data.py --force  # clear transactions & reports, re-seed (keeps users/schemes)
"""

import random
import sys
from datetime import datetime, timedelta, timezone

# Import app and models so we run inside Flask app context
from app import (
    app,
    db,
    Scheme,
    Transaction,
    CorruptionReport,
    corruption_model,
)


# Dummy schemes
SCHEMES = [
    {"name": "Rural Housing Scheme", "allocated_amount": 50_00_000, "description": "Housing for rural families"},
    {"name": "Farm Subsidy Program", "allocated_amount": 1_00_00_000, "description": "Agricultural support"},
    {"name": "Health Insurance Scheme", "allocated_amount": 75_00_000, "description": "Primary health coverage"},
    {"name": "Education Grant", "allocated_amount": 30_00_000, "description": "School and college grants"},
]

# Locations (match training data: CityA, CityB, CityC, TownD, VillageE, Other)
LOCATIONS = ["CityA", "CityB", "CityC", "TownD", "VillageE", "Other"]

# Beneficiary names for variety
BENEFICIARIES = [
    "Ramesh Kumar", "Sunita Devi", "Vijay Singh", "Lakshmi Reddy", "Mohammed Ali",
    "Priya Sharma", "Anil Patel", "Kavitha Nair", "Suresh Gupta", "Meera Iyer",
    "Rajesh Verma", "Padmini Rao", "Krishna Murthy", "Deepa Menon", "Srinivas Joshi",
]


def random_date(days_back=365):
    """Return a random date string in the last days_back days."""
    d = datetime.now(timezone.utc) - timedelta(days=random.randint(0, days_back))
    return d.strftime("%Y-%m-%d")


def seed_schemes(force=False):
    """Create dummy schemes if none exist (or force re-create if force=True)."""
    if force:
        # Remove all transactions and reports first (they reference schemes)
        Transaction.query.delete()
        CorruptionReport.query.delete()
        db.session.commit()
        Scheme.query.delete()
        db.session.commit()
        print("Cleared existing schemes, transactions, and reports.")
    if Scheme.query.count() > 0 and not force:
        print("Schemes already exist, skipping scheme creation.")
        return list(Scheme.query.all())
    for s in SCHEMES:
        scheme = Scheme(
            name=s["name"],
            allocated_amount=s["allocated_amount"],
            description=s.get("description"),
        )
        db.session.add(scheme)
    db.session.commit()
    print(f"Created {len(SCHEMES)} schemes.")
    return Scheme.query.all()


def seed_transactions(schemes, num_transactions=40, force=False):
    """
    Create dummy transactions. Some will have large amount differences or
    high delay so the ML model is likely to flag them as corruption.
    """
    if Transaction.query.count() > 0 and not force:
        print("Transactions already exist, skipping transaction creation.")
        return

    threshold = 0.6
    created = 0
    for _ in range(num_transactions):
        scheme = random.choice(schemes)
        expected = round(random.uniform(10_000, 80_000), 2)
        # ~40% of transactions: large difference or delay to get some "corruption" cases
        if random.random() < 0.4:
            transferred = expected + random.choice([-1, 1]) * random.uniform(15_000, 35_000)
            delay_days = random.randint(45, 100)
        else:
            transferred = expected + random.uniform(-2_000, 2_000)
            delay_days = random.randint(0, 30)
        transferred = round(max(1_000, transferred), 2)
        amount_diff = round(transferred - expected, 2)
        location = random.choice(LOCATIONS)
        beneficiary = random.choice(BENEFICIARIES)

        tx = Transaction(
            scheme_id=scheme.id,
            beneficiary_name=beneficiary,
            expected_amount=expected,
            transferred_amount=transferred,
            amount_difference=amount_diff,
            location=location,
            date=random_date(),
            delay_days=delay_days,
        )
        db.session.add(tx)
        db.session.commit()

        # Run ML model to get corruption probability
        prob = corruption_model.predict_proba(
            scheme_id=scheme.id,
            expected_amount=expected,
            transferred_amount=transferred,
            delay_days=delay_days,
            location=location,
        )
        tx.corruption_probability = prob
        tx.is_corruption_detected = prob >= threshold
        if tx.is_corruption_detected:
            db.session.add(CorruptionReport(
                transaction_id=tx.id,
                scheme_id=tx.scheme_id,
                location=tx.location,
                corruption_probability=prob,
            ))
        db.session.commit()
        created += 1

    print(f"Created {created} transactions with ML corruption predictions.")


def main():
    force = "--force" in sys.argv or "-f" in sys.argv
    with app.app_context():
        db.create_all()
        schemes = seed_schemes(force=force)
        seed_transactions(schemes, force=force)
    print("Done. Start the app with: python app.py")
    print("Open http://127.0.0.1:5000/ and log in with your configured admin account.")
    print("Check Dashboard, Transactions, and Corruption Reports to see the dummy data.")


if __name__ == "__main__":
    main()
