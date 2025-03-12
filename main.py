# main.py
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pandas as pd

from model.ml.data import process_data
from model.ml.model import inference


# Initialize FastAPI app
app = FastAPI()

# Load the model and encoders (assuming they're saved)
try:
    model = joblib.load("model/artifacts/model.joblib")
    encoder = joblib.load("model/artifacts/encoder.joblib")
    lb = joblib.load("model/artifacts/lb.joblib")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    raise


class CensusData(BaseModel):
    age: int
    workclass: Literal["Private", "Public", "Self-employed"]
    fnlgt: int
    education: Literal["HS", "BSc", "MSc", "PhD"]
    education_num: int = Field(alias="education-num")
    marital_status: Optional[str] = Field(alias="marital-status")
    occupation: Optional[str]
    relationship: Optional[str]
    race: Literal["White", "Black", "Asian", "Other"]
    sex: Literal["Male", "Female"]
    capital_gain: Optional[int] = Field(alias="capital-gain", default=0)
    capital_loss: Optional[int] = Field(alias="capital-loss", default=0)
    hours_per_week: Optional[int] = Field(alias="hours-per-week", default=40)
    native_country: Optional[str] = Field(
        alias="native-country", default="United-States"
    )

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "workclass": "Private",
                "fnlgt": 122273,
                "education": "BSc",
                "education-num": 13,
                "marital-status": "Married",
                "occupation": "Prof-specialty",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.get("/")
async def welcome():
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/predict")
async def predict(data: CensusData):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data.dict(by_alias=True)])

        # Apply the same preprocessing as during training
        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        # Transform features
        X, y, _, _ = process_data(
            df,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb,
        )

        # Make prediction
        prediction = inference(model, X)
        probability = model.predict_proba(X)

        # Transform prediction back to original label
        prediction_label = lb.inverse_transform(prediction)[0]

        return {
            "prediction": prediction_label,
            "probability": float(max(probability[0])),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
