import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")
encoder = joblib.load("models/encoder.pkl")

experience = 5
test_score = 75
education = "MCA"

education_encoded = encoder.transform([education])[0]

data = pd.DataFrame({
    "experience": [experience],
    "test_score": [test_score],
    "education": [education_encoded]
})

prediction = model.predict(data)

print("Predicted Salary =", round(prediction[0], 2))