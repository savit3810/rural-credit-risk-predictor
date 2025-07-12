import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# --- Step 1: Create Fake Dataset ---
np.random.seed(42)
n = 100

data = {
    "age": np.random.randint(18, 65, n),
    "income": np.random.randint(10000, 60000, n),
    "land_area": np.round(np.random.uniform(0.5, 5.0, n), 2),
    "rainfall_last_year": np.random.randint(400, 1200, n),
    "market_price": np.random.randint(1000, 2000, n),
    "upi_usage_score": np.round(np.random.uniform(0.0, 1.0, n), 2),
    "crop_type_rice": np.random.randint(0, 2, n),
    "crop_type_sugarcane": np.random.randint(0, 2, n),
    "crop_type_wheat": np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)

# Rule: If income, upi_score, and land are high, more likely to repay
df["loan_repaid"] = (
    (df["income"] > 30000).astype(int)
    & (df["upi_usage_score"] > 0.5).astype(int)
    & (df["land_area"] > 2.0).astype(int)
)

# --- Step 2: Train Model ---
X = df.drop("loan_repaid", axis=1)
y = df["loan_repaid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 3: Save Model ---
joblib.dump(model, "credit_model.pkl")
print("✅ Model trained and saved as credit_model.pkl")
