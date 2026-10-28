from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Email Classifier API")

# Load the entire pipeline (including TF-IDF)
model = joblib.load('model.joblib')

class EmailInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    label: str      # "spam" or "ham"
    confidence: float

@app.post("/predict", response_model=PredictionOutput)
def predict(email: EmailInput):
    # Predict directly on raw text
    pred = model.predict([email.text])[0]
    proba = model.predict_proba([email.text])[0]
    label = "spam" if pred == 1 else "ham"
    confidence = float(proba[pred])
    return PredictionOutput(label=label, confidence=confidence)

@app.get("/health")
def health():
    return {"status": "ok"}