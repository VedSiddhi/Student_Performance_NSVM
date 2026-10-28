
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Generate reproducible demonstration data
rng = np.random.default_rng(42)
n = 1200

study_hours = rng.uniform(0, 10, n)
attendance = rng.uniform(40, 100, n)
previous_marks = rng.uniform(20, 100, n)
assignment_score = rng.uniform(20, 100, n)
sleep_hours = rng.uniform(3, 10, n)
classes_missed = rng.integers(0, 31, n)

# Create an illustrative score with a non-linear relationship
score = (
    2.5 * study_hours
    + 0.20 * attendance
    + 0.20 * previous_marks
    + 0.12 * assignment_score
    + 1.2 * sleep_hours
    - 0.45 * classes_missed
    + 0.15 * study_hours * (attendance / 10)
    + rng.normal(0, 5, n)
)

# Define demonstration categories
performance = np.where(
    score < 32, "Needs Improvement",
    np.where(score < 48, "Average", "Good")
)

data = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "previous_marks": previous_marks,
    "assignment_score": assignment_score,
    "sleep_hours": sleep_hours,
    "classes_missed": classes_missed,
    "performance": performance
})

data.to_csv("student_data.csv", index=False)

X = data.drop("performance", axis=1)
y = data["performance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf", C=1.0, gamma="scale")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Non-Linear SVM Performance")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n",
      confusion_matrix(y_test, predictions,
                       labels=model.classes_))
print("\nClassification Report:\n",
      classification_report(y_test, predictions,
                            zero_division=0))

with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nDataset, model and scaler saved successfully.")
print("Class order:", model.classes_)
