import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("Starting improved training pipeline...")

# Load dataset
data_path = os.path.join("data", "students.csv")
df = pd.read_csv(data_path)

# Basic validation
required_columns = {"study_hours", "attendance", "previous_score", "pass"}
if not required_columns.issubset(df.columns):
    raise ValueError("Dataset missing required columns")

# Features and target
X = df.drop("pass", axis=1)
y = df["pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ML pipeline (scaling + model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save pipeline (scaler + model together)
model_path = os.path.join("model", "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print("Improved model saved successfully")
