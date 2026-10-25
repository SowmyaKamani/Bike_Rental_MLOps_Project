import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Bike Rental Demand Prediction API",
    description="API for predicting bike rental demand",
    version="1.0.0"
)


# ============================================================
# MLflow configuration
# ============================================================

mlflow.set_tracking_uri("sqlite:///../mlflow.db")

MODEL_NAME = "Bike_Rental_Demand_Model"

model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@champion"
)

print("Champion model loaded successfully!")


# ============================================================
# Input schema
# ============================================================

class BikeRentalInput(BaseModel):

    DayOfWeek: str
    Season: str
    Weather: str

    Temperature: float
    Humidity: float
    WindSpeed: float
    RentalDuration: float

    UserType: str
    PaymentMethod: str
    Cancelled: int

    Year: int
    Month: int
    Day: int
    DayOfWeek_Num: int
    Hour: int
    Is_Weekend: int

    Month_Sin: float
    Month_Cos: float
    Hour_Sin: float
    Hour_Cos: float


# ============================================================
# Home endpoint
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Bike Rental Demand API is running"
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(data: BikeRentalInput):

    input_data = data.model_dump()

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    return {
        "prediction": float(prediction[0])
    }