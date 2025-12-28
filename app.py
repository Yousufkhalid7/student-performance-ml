from flask import Flask, render_template, request
import pickle
import os
import numpy as np

app = Flask(__name__)

# Load trained ML pipeline
model_path = os.path.join("model", "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_score = float(request.form["previous_score"])

        features = np.array([[study_hours, attendance, previous_score]])
        prediction = model.predict(features)[0]

        result = "Pass" if prediction == 1 else "Fail"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Invalid input. Please enter valid numbers."
        )

if __name__ == "__main__":
    app.run(debug=True)
