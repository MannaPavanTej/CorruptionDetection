# How the ML Model Works - Detailed Explanation

## Overview
The Corruption Detection System uses a **Random Forest Classifier** (from scikit-learn) to predict whether a transaction is likely to be corrupt based on transaction features.

---

## 1. Training Phase (`model_training.py`)

### Step 1: Generate Synthetic Training Data
The script creates **300 dummy transactions** with these features:
- **scheme_id**: Which government scheme (1-5)
- **expected_amount**: Amount that should be transferred (₹5,000 - ₹100,000)
- **transferred_amount**: Amount actually transferred (with noise and fraud patterns)
- **amount_difference**: `transferred_amount - expected_amount`
- **delay_days**: Days of delay (0-120)
- **location**: One of: CityA, CityB, CityC, TownD, VillageE, Other

### Step 2: Create Labels (Corruption or Not)
Each transaction gets a **label** (0 = Normal, 1 = Corruption) based on simple rules:
```python
# Corruption is flagged if ANY of these are true:
- amount_difference > ₹15,000  (too much transferred)
- amount_difference < -₹10,000  (too little transferred)
- delay_days > 60               (very delayed)
```

### Step 3: Encode Categorical Data
- **Location** (text) → **location_encoded** (number) using `LabelEncoder`
  - CityA → 0, CityB → 1, CityC → 2, TownD → 3, VillageE → 4, Other → 5

### Step 4: Prepare Features for ML
The model uses these **6 numeric features**:
1. `scheme_id` (1-5)
2. `expected_amount` (e.g., 50000.0)
3. `transferred_amount` (e.g., 65000.0)
4. `amount_difference` (e.g., 15000.0)
5. `delay_days` (e.g., 75)
6. `location_encoded` (e.g., 0 for CityA)

### Step 5: Train Random Forest
- **Algorithm**: Random Forest (ensemble of 50 decision trees)
- **Input**: Feature matrix X (300 rows × 6 columns), Labels y (300 values: 0 or 1)
- **Output**: Trained model that can predict corruption probability
- **Saved**: Model + encoder saved as `models/corruption_model.pkl`

---

## 2. Prediction Phase (When You Add a Transaction)

### Step 1: User Submits Transaction Form
You enter:
- Scheme: "Rural Housing Scheme" (ID: 1)
- Expected Amount: ₹50,000
- Transferred Amount: ₹70,000
- Location: "CityA"
- Delay Days: 80

### Step 2: Calculate Features
```python
amount_difference = 70000 - 50000 = 20000
scheme_id = 1
expected_amount = 50000.0
transferred_amount = 70000.0
delay_days = 80
location = "CityA" → location_encoded = 0 (via LabelEncoder)
```

### Step 3: Call ML Model
```python
features = [1, 50000.0, 70000.0, 20000.0, 80, 0]
probability = model.predict_proba([features])[0][1]
# Returns: 0.85 (85% probability of corruption)
```

### Step 4: Decision Logic
```python
threshold = 0.6  # 60%
if probability >= 0.6:
    is_corruption_detected = True
    # Create entry in corruption_reports table
else:
    is_corruption_detected = False
```

### Step 5: Store Results
- Transaction saved with:
  - `corruption_probability = 0.85` (85%)
  - `is_corruption_detected = True`
- Corruption Report created (if flagged)

---

## 3. How Random Forest Makes Predictions

### What is Random Forest?
- **Ensemble method**: Combines predictions from **50 decision trees**
- Each tree votes: "corruption" or "normal"
- Final probability = percentage of trees voting "corruption"

### Example Decision Tree Logic:
```
Tree 1 might ask:
  - Is amount_difference > 15000? → YES → corruption
  - Is delay_days > 60? → YES → corruption
  
Tree 2 might ask:
  - Is transferred_amount > expected_amount * 1.3? → YES → corruption
  
Tree 3 might ask:
  - Is location = CityA AND delay > 50? → YES → corruption
```

**Final Prediction**: If 42 out of 50 trees vote "corruption" → probability = 42/50 = 0.84 (84%)

---

## 4. Why This Model Flags Corruption

The model learned patterns from training data:

### High Corruption Probability When:
1. **Large positive amount difference** (₹15,000+ more transferred than expected)
2. **Large negative amount difference** (₹10,000+ less transferred)
3. **Long delays** (60+ days)
4. **Combinations**: e.g., medium difference + long delay + certain locations

### Low Corruption Probability When:
1. **Small amount differences** (±₹2,000)
2. **Short delays** (< 30 days)
3. **Normal patterns**: Expected ≈ Transferred, reasonable delays

---

## 5. Dashboard Display

### How Fraud % is Calculated:
- **Fraud %** = `corruption_probability × 100`
- Example: If model returns `0.73` → Dashboard shows **73%**

### How "Corruption Detected" Status Works:
- **Red Badge** (Corruption Detected): `probability >= 0.6`
- **Green Badge** (Normal): `probability < 0.6`

### Corruption by Location Chart:
- Groups transactions by `location`
- Counts how many have `is_corruption_detected = True` per location
- Displays as bar chart using Chart.js

---

## 6. Important Notes

### Model Limitations:
1. **Trained on synthetic data** - Real-world patterns may differ
2. **Threshold is 60%** - Can be adjusted (lower = more sensitive, higher = stricter)
3. **Location encoding** - Unknown locations default to "Other"

### Model Strengths:
1. **Fast predictions** - No need to retrain for each transaction
2. **Handles multiple features** - Considers amount, delay, location together
3. **Probability output** - Not just yes/no, but confidence level

---

## 7. Example Scenarios

### Scenario 1: Normal Transaction
```
Expected: ₹50,000
Transferred: ₹50,500
Difference: ₹500
Delay: 5 days
Location: CityB

Model Output: 0.15 (15%)
Result: Normal (below 60% threshold)
```

### Scenario 2: Suspicious Transaction
```
Expected: ₹50,000
Transferred: ₹75,000
Difference: ₹25,000
Delay: 90 days
Location: CityA

Model Output: 0.92 (92%)
Result: Corruption Detected (above 60% threshold)
```

### Scenario 3: Borderline Case
```
Expected: ₹50,000
Transferred: ₹62,000
Difference: ₹12,000
Delay: 45 days
Location: TownD

Model Output: 0.58 (58%)
Result: Normal (just below threshold)
```

---

## Summary

**The ML model works by:**
1. Learning patterns from 300 training examples
2. Using 6 features (scheme, amounts, difference, delay, location)
3. Predicting corruption probability (0-100%)
4. Flagging transactions with probability ≥ 60%
5. Storing results in database for dashboard display

**The dashboard shows:**
- Total corruption cases (count of flagged transactions)
- Fraud % per transaction (from ML model)
- Corruption by location (grouped counts)
- Status badges (Normal vs Corruption Detected)
