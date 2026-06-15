"""
model_training.py
------------------

This script creates a **synthetic dataset** of government scheme transactions,
trains a RandomForestClassifier to detect potential corruption, and saves:

- A trained RandomForest model
- A fitted LabelEncoder for locations

Both are stored together in a single pickle file under the `models` directory.

You can later replace the synthetic data with a real dataset while keeping
the same feature-engineering and saving logic.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


def generate_synthetic_data(n_samples: int = 300) -> pd.DataFrame:
    """
    Generate a synthetic dataset that roughly mimics government scheme
    transactions with a simple rule-based corruption label.

    Features:
    - scheme_id (categorical-like integer)
    - expected_amount
    - transferred_amount
    - amount_difference
    - delay_days
    - location (categorical; includes 'Other' to handle unseen cities)
    - label (0 = normal, 1 = corruption suspected)
    """
    rng = np.random.default_rng(seed=42)

    scheme_ids = rng.integers(1, 6, size=n_samples)  # 5 different schemes
    expected_amount = rng.integers(5_000, 100_000, size=n_samples)

    # Simulate transferred amount with some noise and systematic "fraud"
    noise = rng.normal(loc=0, scale=3_000, size=n_samples)
    fraud_factor = rng.integers(-30_000, 30_000, size=n_samples)
    transferred_amount = expected_amount + noise + fraud_factor
    transferred_amount = np.clip(transferred_amount, 1_000, 150_000)

    delay_days = rng.integers(0, 120, size=n_samples)

    locations = np.array(
        [
            "CityA",
            "CityB",
            "CityC",
            "TownD",
            "VillageE",
            "Other",  # reserved for unseen locations at prediction time
        ]
    )
    location = rng.choice(locations, size=n_samples, replace=True)

    amount_difference = transferred_amount - expected_amount

    # Heuristic: label as "corruption" if large difference and/or big delay,
    # plus some randomness to avoid a perfectly deterministic dataset.
    label = (
        (
            (amount_difference > 15_000)
            | (amount_difference < -10_000)
            | (delay_days > 60)
        )
        .astype(int)
    )

    # Add mild noise: flip some labels
    flip_mask = rng.random(n_samples) < 0.05
    label = np.where(flip_mask, 1 - label, label)

    df = pd.DataFrame(
        {
            "scheme_id": scheme_ids,
            "expected_amount": expected_amount,
            "transferred_amount": transferred_amount,
            "amount_difference": amount_difference,
            "delay_days": delay_days,
            "location": location,
            "label": label,
        }
    )
    return df


def train_and_save_model():
    """Train a RandomForest model and save it with the location encoder."""
    print("Generating synthetic training data...")
    df = generate_synthetic_data()

    # Encode location using LabelEncoder
    location_encoder = LabelEncoder()
    df["location_encoded"] = location_encoder.fit_transform(df["location"])

    feature_cols = [
        "scheme_id",
        "expected_amount",
        "transferred_amount",
        "amount_difference",
        "delay_days",
        "location_encoded",
    ]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=1,
    )

    print("Training RandomForestClassifier...")
    model.fit(X_train, y_train)

    print("Evaluating model on test split...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_bundle = {
        "model": model,
        "location_encoder": location_encoder,
        "feature_columns": feature_cols,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    model_path = os.path.join(models_dir, "corruption_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"Model and encoder saved to: {model_path}")


if __name__ == "__main__":
    train_and_save_model()

