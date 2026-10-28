from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

app = FastAPI(
    title="Spam Email Classifier API",
    description="Production-ready FastAPI service for classifying emails as spam or ham using a scikit-learn TF-IDF + Logistic Regression pipeline.",
    version="1.0.0"
)

# Load the trained machine learning pipeline
model = joblib.load("model.joblib")


class EmailInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="The content of the email or message to classify",
        json_schema_extra={"example": "Congratulations! You won a $1,000 Walmart gift card. Click here."}
    )


class PredictionOutput(BaseModel):
    label: str = Field(description="Predicted classification label: 'spam' or 'ham'")
    confidence: float = Field(description="Prediction confidence score between 0.0 and 1.0")


@app.get("/", tags=["General"])
def root():
    """Root endpoint providing service metadata and documentation links."""
    return {
        "service": "Spam Email Classifier API",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.get("/health", tags=["General"])
def health():
    """Health check endpoint for Docker container and CI/CD pipelines."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
def predict(email: EmailInput):
    """
    Predict whether an email text is spam or ham (legitimate).
    Accepts raw text and passes it through the TF-IDF + Logistic Regression pipeline.
    """
    if not email.text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty or whitespace.")

    pred = int(model.predict([email.text])[0])
    proba = model.predict_proba([email.text])[0]
    label = "spam" if pred == 1 else "ham"
    confidence = float(proba[pred])

    return PredictionOutput(label=label, confidence=confidence)