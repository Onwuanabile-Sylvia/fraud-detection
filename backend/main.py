from pathlib import Path
import os

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from openai import OpenAI

from backend.explain import get_top_features


# ==================================================
# APPLICATION
# ==================================================

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description=(
        "API for predicting fraudulent credit card "
        "transactions using an XGBoost model."
    ),
    version="1.0.0"
)


# ==================================================
# FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
THRESHOLD_PATH = BASE_DIR / "fraud_threshold.pkl"


# ==================================================
# LOAD TRAINED ARTIFACTS
# ==================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))


# ==================================================
# FEATURE ORDER
# ==================================================

FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ==================================================
# OPENAI CLIENT
# ==================================================

openai_client = None

if os.getenv("OPENAI_API_KEY"):
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )


# ==================================================
# INPUT VALIDATION
# ==================================================

class Transaction(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    Time: float = Field(
        ...,
        ge=0,
        description=(
            "Seconds elapsed between this transaction "
            "and the first transaction."
        )
    )

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount."
    )


# ==================================================
# PREPROCESSING
# ==================================================

def preprocess_transaction(
    transaction_data: dict
) -> pd.DataFrame:

    transaction_df = pd.DataFrame(
        [transaction_data]
    )

    transaction_df = transaction_df[
        FEATURE_COLUMNS
    ].copy()

    transaction_df[
        ["Time", "Amount"]
    ] = scaler.transform(
        transaction_df[
            ["Time", "Amount"]
        ]
    )

    return transaction_df


# ==================================================
# LLM EXPLANATION
# ==================================================

def generate_llm_explanation(
    prediction,
    probability,
    threshold,
    top_features
):

    # ----------------------------------------------
    # Fallback if LLM is unavailable
    # ----------------------------------------------

    if openai_client is None:

        if prediction == 1:
            return (
                "The transaction was flagged because "
                "the model estimated a high fraud risk "
                "based on the most influential features."
            )

        return (
            "The transaction appears legitimate because "
            "the model estimated a fraud probability below "
            "the decision threshold."
        )


    # ----------------------------------------------
    # Prepare ONLY model explanation information
    # ----------------------------------------------

    feature_summary = []

    for item in top_features:

        feature_summary.append(
            (
                f"{item['feature']}: "
                f"{item['direction']}"
            )
        )

    feature_text = "\n".join(
        feature_summary
    )


    # ----------------------------------------------
    # LLM instructions
    # ----------------------------------------------

    instructions = """
You are an explanation assistant for a fraud
detection system.

Your ONLY job is to convert the supplied model
prediction and SHAP feature information into a
short, clear explanation for a non-technical user.

IMPORTANT RULES:

1. Do NOT make your own fraud decision.
2. Do NOT change the model's prediction.
3. Do NOT invent information.
4. Do NOT mention transaction values.
5. Do NOT request or infer raw transaction data.
6. Explain only the information supplied.
7. Keep the explanation to 1-2 sentences.
8. Use cautious language such as "the model"
   or "the system".
"""


    # ----------------------------------------------
    # Information supplied to the LLM
    # ----------------------------------------------

    user_input = f"""
Model prediction: {prediction}

Model probability of fraud: {probability}

Decision threshold: {threshold}

Top SHAP feature contributions:

{feature_text}
"""


    try:

        response = openai_client.responses.create(
            model="gpt-5.6-luna",
            instructions=instructions,
            input=user_input
        )

        explanation = response.output_text.strip()

        if explanation:
            return explanation

        raise ValueError(
            "LLM returned an empty explanation."
        )


    except Exception:

        # ------------------------------------------
        # LLM failure must NOT stop prediction
        # ------------------------------------------

        if prediction == 1:

            return (
                "The transaction was flagged because "
                "the model estimated a high fraud risk "
                "based on the most influential features."
            )

        return (
            "The transaction appears legitimate because "
            "the model estimated a fraud probability "
            "below the decision threshold."
        )


# ==================================================
# ROOT ENDPOINT
# ==================================================

@app.get("/")
def home():

    return {
        "message": "Fraud Detection API is running",
        "model": "XGBoost",
        "threshold": threshold,
        "features": len(FEATURE_COLUMNS)
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "llm_configured": openai_client is not None
    }


# ==================================================
# PREDICTION ENDPOINT
# ==================================================

@app.post("/predict")
def predict(transaction: Transaction):

    try:

        # ------------------------------------------
        # 1. Get validated input
        # ------------------------------------------

        transaction_data = transaction.model_dump()


        # ------------------------------------------
        # 2. Apply training preprocessing
        # ------------------------------------------

        processed_transaction = (
            preprocess_transaction(
                transaction_data
            )
        )


        # ------------------------------------------
        # 3. Generate fraud probability
        # ------------------------------------------

        probability = float(
            model.predict_proba(
                processed_transaction
            )[0][1]
        )


        # ------------------------------------------
        # 4. Apply saved decision threshold
        # ------------------------------------------

        prediction = int(
            probability >= threshold
        )


        # ------------------------------------------
        # 5. Create prediction message
        # ------------------------------------------

        if prediction == 1:

            message = (
                "Fraudulent transaction detected"
            )

        else:

            message = (
                "Transaction appears legitimate"
            )


        # ------------------------------------------
        # 6. Generate SHAP explanation
        # ------------------------------------------

        top_features = get_top_features(
            model=model,
            transaction=processed_transaction,
            feature_names=FEATURE_COLUMNS,
            top_n=5
        )


        # ------------------------------------------
        # 7. Generate plain-English explanation
        # ------------------------------------------

        explanation = generate_llm_explanation(
            prediction=prediction,
            probability=probability,
            threshold=threshold,
            top_features=top_features
        )


        # ------------------------------------------
        # 8. Return final response
        # ------------------------------------------

        return {
            "prediction": prediction,
            "probability": round(
                probability,
                6
            ),
            "threshold": threshold,
            "message": message,
            "top_features": top_features,
            "explanation": explanation
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed",
                "error": str(error)
            }
        )