import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

print("Starting training script...")

# Load dataset
data_path = os.path.join("data", "students.csv")
df = pd.read_csv(data_path)

print("Dataset loaded")

# Features and target
X = df.drop("pass", axis=1)
y = df["pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Save model
model_path = os.path.join("model", "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully")
