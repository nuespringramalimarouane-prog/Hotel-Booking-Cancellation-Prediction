import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Hotel Booking Cancellation API", version="1.0.0")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "hotel_booking_final_pipeline.joblib"


class BookingBatch(BaseModel):
    bookings: list[dict[str, Any]] = Field(min_length=1)


@lru_cache(maxsize=1)
def load_artifact():
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    required = {
        "stays_in_weekend_nights", "stays_in_week_nights", "adults",
        "children", "babies", "previous_cancellations", "adr",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required booking fields: {sorted(missing)}")

    data["total_nights"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    data["total_guests"] = data["adults"] + data["children"].fillna(0) + data["babies"]
    data["has_children"] = (data["children"].fillna(0) > 0).astype(int)
    data["has_babies"] = (data["babies"] > 0).astype(int)
    data["has_previous_cancellations"] = (data["previous_cancellations"] > 0).astype(int)
    data["adr_per_guest"] = data["adr"] / data["total_guests"].replace(0, float("nan"))
    return data


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(batch: BookingBatch):
    try:
        artifact = load_artifact()
        engineered = engineer_features(pd.DataFrame(batch.bookings))
        features = engineered.drop(columns=artifact["drop_columns"], errors="ignore")
        features = features[artifact["feature_columns"]]
        transformed = artifact["preprocessor"].transform(features)
        probabilities = artifact["model"].predict_proba(transformed)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        return {
            "predictions": [
                {"is_canceled": int(prediction), "cancellation_probability": float(probability)}
                for prediction, probability in zip(predictions, probabilities)
            ]
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
