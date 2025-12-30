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

	if study_hours < 0 or study_hours > 24:
		raise ValueError("Study hours must be between 0 and 24")

	if attendance < 0 or attendance > 100:
		raise ValueError("Attendance must be between 0 and 100")

	if previous_score < 0 or previous_score > 100:
		raise ValueError("Previous score must be between 0 and 100")

        features = np.array([[study_hours, attendance, previous_score]])
        prediction = model.predict(features)[0]

        result = "Pass" if prediction == 1 else "Fail"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {result}"
        )
	
    except ValueError as ve:
	return render_template(
		"index.html",
		prediction_text=f"Input error: {str(ve)}"
	    
	
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Invalid input. Please enter valid numbers."
        )

if __name__ == "__main__":
    app.run(debug=True)
