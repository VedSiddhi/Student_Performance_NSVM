
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            values = {
                "study_hours": float(request.form["study_hours"]),
                "attendance": float(request.form["attendance"]),
                "previous_marks": float(request.form["previous_marks"]),
                "assignment_score": float(request.form["assignment_score"]),
                "sleep_hours": float(request.form["sleep_hours"]),
                "classes_missed": float(request.form["classes_missed"])
            }

            limits = {
                "study_hours": (0, 24),
                "attendance": (0, 100),
                "previous_marks": (0, 100),
                "assignment_score": (0, 100),
                "sleep_hours": (0, 24),
                "classes_missed": (0, 365)
            }

            for key, value in values.items():
                low, high = limits[key]
                if not np.isfinite(value) or not low <= value <= high:
                    raise ValueError(
                        f"{key.replace('_', ' ').title()} "
                        f"must be between {low} and {high}."
                    )

            if values["classes_missed"] != int(values["classes_missed"]):
                raise ValueError("Classes missed must be a whole number.")

            features = np.array([list(values.values())])
            scaled = scaler.transform(features)
            result = model.predict(scaled)[0]

        except (ValueError, KeyError) as exc:
            error = str(exc) or "Please enter valid numerical values."

    return render_template(
        "index.html",
        result=result,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)
