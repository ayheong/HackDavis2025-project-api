from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ORDERS = {
    "heart_disease": (0, 2, 0),
    "stroke": (1, 0, 1),
    "diabetes": (1, 0, 1),
    "suicide": (0, 1, 0),
    "cancer": (0, 1, 1)
}

class ForecastRequest(BaseModel):
    data: List[float]
    model_name: str

class ForecastResponse(BaseModel):
    forecast: float

@app.post("/forecast", response_model=ForecastResponse)
def forecast_next_value(request: ForecastRequest):
    order = MODEL_ORDERS.get(request.model_name)
    if order is None:
        return {"forecast": request.data[-1]}  # fallback

    if len(request.data) < max(order):
        return {"forecast": request.data[-1]}

    series = pd.Series(request.data)
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    next_forecast = model_fit.forecast(steps=1).iloc[0]
    return ForecastResponse(forecast=round(float(next_forecast), 2))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
