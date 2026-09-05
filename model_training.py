"""
model_training.py
-----------------

Creates a synthetic government-transaction dataset, trains a
RandomForestClassifier to estimate corruption risk, evaluates the model
using multiple metrics, and saves the trained model bundle.

NOTE:
The dataset is synthetic and intended for demonstration/prototyping.
It should be replaced with a real, properly labelled dataset for
real-world deployment.
"""

import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# -------------------------------------------------------------------
# 1. Generate synthetic transaction data
# -------------------------------------------------------------------

def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic government scheme transaction data.

    Features:
    - scheme_id
    - expected_amount
    - transferred_amount
    - amount_difference
    - delay_days
    - location
    - label (0 = normal, 1 = corruption suspected)

    The corruption label represents several simulated risk patterns
    rather than a single simple threshold.
    """

    rng = np.random.default_rng(seed=42)

    # Government schemes
    scheme_ids = rng.integers(1, 6, size=n_samples)

    # Expected transaction amount
    expected_amount = rng.integers(
        5_000,
        100_000,
        size=n_samples
    )

    # Normal transaction variation
    normal_noise = rng.normal(
        loc=0,
        scale=2_500,
        size=n_samples
    )

    transferred_amount = expected_amount + normal_noise

    # Simulate different transaction risk patterns
    risk_pattern = rng.choice(
        ["normal", "overpayment", "underpayment", "delayed"],
        size=n_samples,
        p=[0.70, 0.10, 0.08, 0.12]
    )

    # Overpayment pattern
    overpayment_mask = risk_pattern == "overpayment"
    transferred_amount[overpayment_mask] += rng.uniform(
        15_000,
        35_000,
        size=overpayment_mask.sum()
    )

    # Underpayment pattern
    underpayment_mask = risk_pattern == "underpayment"
    transferred_amount[underpayment_mask] -= rng.uniform(
        10_000,
        25_000,
        size=underpayment_mask.sum()
    )

    # Keep values realistic
    transferred_amount = np.clip(
        transferred_amount,
        1_000,
        150_000
    )

    # Transaction delay
    delay_days = rng.integers(
        0,
        120,
        size=n_samples
    )

    # Locations
    locations = np.array(
        [
            "CityA",
            "CityB",
            "CityC",
            "TownD",
            "VillageE",
            "Other",
        ]
    )

    location = rng.choice(
        locations,
        size=n_samples,
        replace=True
    )

    amount_difference = (
        transferred_amount - expected_amount
    )

    # ----------------------------------------------------------------
    # Simulate corruption risk using multiple behavioural patterns
    # ----------------------------------------------------------------

    risk_score = np.zeros(n_samples)

    # Large positive discrepancy
    risk_score += (
        amount_difference > 15_000
    ) * 2

    # Large negative discrepancy
    risk_score += (
        amount_difference < -10_000
    ) * 1.5

    # Long delays
    risk_score += (
        delay_days > 60
    ) * 1.5

    # Very large transactions receive additional risk
    risk_score += (
        transferred_amount > 100_000
    ) * 1

    # Combination of high amount + delay
    risk_score += (
        (transferred_amount > 80_000)
        & (delay_days > 45)
    ) * 1.5

    # Convert risk score into corruption label
    label = (
        risk_score >= 2
    ).astype(int)

    # Add small amount of label noise
    flip_mask = rng.random(n_samples) < 0.03
    label = np.where(
        flip_mask,
        1 - label,
        label
    )

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


# -------------------------------------------------------------------
# 2. Train and evaluate model
# -------------------------------------------------------------------

def train_and_save_model():

    print("=" * 60)
    print("CORRUPTION DETECTION MODEL TRAINING")
    print("=" * 60)
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "models")

    evaluation_dir = os.path.join(base_dir, "evaluation")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)

    print("\nGenerating synthetic dataset...")

    df = generate_synthetic_data()

    print(f"Dataset size: {len(df)} transactions")
    print(f"Normal transactions: {(df['label'] == 0).sum()}")
    print(f"Suspicious transactions: {(df['label'] == 1).sum()}")

    # ---------------------------------------------------------------
    # Encode location
    # ---------------------------------------------------------------

    location_encoder = LabelEncoder()

    df["location_encoded"] = location_encoder.fit_transform(
        df["location"]
    )

    # ---------------------------------------------------------------
    # Features
    # ---------------------------------------------------------------

    feature_cols = [
        "scheme_id",
        "expected_amount",
        "transferred_amount",
        "amount_difference",
        "delay_days",
        "location_encoded",
    ]

    X = df[feature_cols]
    y = df["label"]

    # ---------------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # ---------------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Normal",
                "Corruption Suspected"
            ]
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

# Confusion Matrix Visualization
    plt.figure(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=[
            "Normal",
            "Corruption Suspected"
        ],
        #cmap="Blues"
    )

    plt.title("Corruption Detection - Confusion Matrix")
    plt.tight_layout()

    confusion_path = os.path.join(
        evaluation_dir,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300
    )

    plt.close()

    print(
        f"Confusion matrix saved to: {confusion_path}"
    )

    # ROC Curve
    plt.figure(figsize=(7, 6))

    RocCurveDisplay.from_predictions(
        y_test,
        y_probability
    )

    plt.title(
        f"ROC Curve (AUC = {roc_auc:.4f})"
    )

    plt.tight_layout()

    roc_path = os.path.join(
        evaluation_dir,
        "roc_curve.png"
    )

    plt.savefig(
        roc_path,
        dpi=300
    )

    plt.close()

    print(
        f"ROC curve saved to: {roc_path}"
    )
    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------

    print("\nFeature Importance:")

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        by="importance",
        ascending=False
    )
    # Feature Importance Visualization
    plt.figure(figsize=(9, 6))

    plt.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    importance_path = os.path.join(
        evaluation_dir,
        "feature_importance.png"
    )

    plt.savefig(
        importance_path,
        dpi=300
    )

    plt.close()

    print(
        f"Feature importance chart saved to: {importance_path}"
    )
    for _, row in importance_df.iterrows():
        print(
            f"{row['feature']:<25} "
            f"{row['importance']:.4f}"
        )

    # ---------------------------------------------------------------
    # Save model
    # ---------------------------------------------------------------
    

    model_bundle = {
        "model": model,
        "location_encoder": location_encoder,
        "feature_columns": feature_cols,
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_version": "2.0",
    }

    model_path = os.path.join(
        models_dir,
        "corruption_model.pkl"
    )

    with open(
        model_path,
        "wb"
    ) as f:
        pickle.dump(
            model_bundle,
            f
        )

    print(
        f"\nModel saved successfully to:\n"
        f"{model_path}"
    )

    print("\nTraining completed successfully.")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    train_and_save_model()
