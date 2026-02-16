"""
AQI Forecast API
FastAPI endpoint for serving predictions
AQI Scale: 1-5 (OpenWeather)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import os

from src.feature_store import AQIFeatureStore

app = FastAPI(
    title="AQI Prediction API",
    description="Real-time Air Quality Index forecasting for Karachi",
    version="2.0.0"
)


class AQIResponse(BaseModel):
    """Response model for current AQI"""
    location: str
    timestamp: datetime
    aqi: float
    aqi_category: str
    pm2_5: float
    pm10: float
    temperature: float
    humidity: float
    wind_speed: float
    scale: str = "1-5 (OpenWeather)"


class ForecastResponse(BaseModel):
    """Response model for forecast"""
    horizon: str
    forecast_time: datetime
    predicted_aqi: float
    aqi_category: str
    confidence: Optional[str] = None
    model_name: str
    mae: float
    r2: float


def get_aqi_category(aqi: float) -> str:
    """Get AQI category from value"""
    if aqi <= 1.5:
        return "Good"
    elif aqi <= 2.5:
        return "Fair"
    elif aqi <= 3.5:
        return "Moderate"
    elif aqi <= 4.5:
        return "Poor"
    else:
        return "Very Poor"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer ALL features to match training pipeline
    Creates 82 features used by optimized models
    """
    # Time features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 10) | 
                          (df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'pm2_5_lag_{lag}h'] = df['pm2_5'].shift(lag)
        df[f'pm10_lag_{lag}h'] = df['pm10'].shift(lag)
        df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)
    
    # Rolling features
    for window in [3, 6, 12, 24, 48]:
        df[f'pm2_5_rolling_mean_{window}h'] = df['pm2_5'].rolling(window, min_periods=1).mean()
        df[f'pm2_5_rolling_std_{window}h'] = df['pm2_5'].rolling(window, min_periods=1).std()
        df[f'pm2_5_rolling_max_{window}h'] = df['pm2_5'].rolling(window, min_periods=1).max()
        df[f'pm2_5_rolling_min_{window}h'] = df['pm2_5'].rolling(window, min_periods=1).min()
        df[f'aqi_rolling_mean_{window}h'] = df['aqi'].rolling(window, min_periods=1).mean()
        df[f'aqi_rolling_std_{window}h'] = df['aqi'].rolling(window, min_periods=1).std()
    
    # Rate of change features
    df['pm2_5_change_1h'] = df['pm2_5'].diff(1)
    df['pm2_5_change_3h'] = df['pm2_5'].diff(3)
    df['pm2_5_change_6h'] = df['pm2_5'].diff(6)
    
    # Interaction features
    df['pm2_5_x_wind'] = df['pm2_5'] * df['wind_speed']
    df['pm2_5_x_humidity'] = df['pm2_5'] * df['humidity']
    df['pm2_5_x_temp'] = df['pm2_5'] * df['temperature']
    df['wind_x_humidity'] = df['wind_speed'] * df['humidity']
    df['temp_x_humidity'] = df['temperature'] * df['humidity']
    df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1)
    
    # Weather change features
    df['temp_change'] = df['temperature'].diff(1)
    df['wind_change'] = df['wind_speed'].diff(1)
    df['pressure_change'] = df['pressure'].diff(1)
    
    return df


def load_models() -> Dict:
    """Load all trained models"""
    models = {}
    
    for horizon in ['24h', '48h', '72h']:
        model_path = f'models/aqi_model_{horizon}.joblib'
        metadata_path = f'models/aqi_model_{horizon}_metadata.joblib'
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            models[horizon] = {
                'model': joblib.load(model_path),
                'metadata': joblib.load(metadata_path)
            }
    
    return models


@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "service": "AQI Prediction API",
        "version": "2.0.0",
        "location": "Karachi, Pakistan",
        "aqi_scale": "1-5 (OpenWeather European Index)",
        "endpoints": [
            "/current - Current air quality",
            "/forecast/{horizon} - Forecast (24h, 48h, 72h)",
            "/forecast/all - All forecasts",
            "/health - API health check",
            "/history?days=7 - Historical data"
        ]
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint"""
    with AQIFeatureStore() as fs:
        try:
            count = fs.raw_features.count_documents({})
            latest = fs.raw_features.find_one(sort=[('timestamp', -1)])
            
            models = load_models()
            
            return {
                "status": "healthy",
                "database": "connected",
                "data_records": count,
                "latest_data": latest['timestamp'] if latest else None,
                "models_loaded": len(models),
                "available_horizons": list(models.keys())
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


@app.get("/current", response_model=AQIResponse, tags=["Data"])
async def get_current_aqi():
    """Get current air quality data"""
    with AQIFeatureStore() as fs:
        latest = fs.raw_features.find_one(sort=[('timestamp', -1)])
        
        if not latest:
            raise HTTPException(status_code=404, detail="No data available")
        
        return AQIResponse(
            location=latest.get('location', 'Karachi'),
            timestamp=latest['timestamp'],
            aqi=latest['aqi'],
            aqi_category=get_aqi_category(latest['aqi']),
            pm2_5=latest['pm2_5'],
            pm10=latest['pm10'],
            temperature=latest['temperature'],
            humidity=latest['humidity'],
            wind_speed=latest['wind_speed']
        )


@app.get("/forecast/{horizon}", response_model=ForecastResponse, tags=["Forecast"])
async def get_forecast(horizon: str):
    """
    Get AQI forecast for specific horizon
    
    Args:
        horizon: Forecast horizon ('24h', '48h', or '72h')
    """
    if horizon not in ['24h', '48h', '72h']:
        raise HTTPException(
            status_code=400, 
            detail="Invalid horizon. Use '24h', '48h', or '72h'"
        )
    
    # Load model
    model_path = f'models/aqi_model_{horizon}.joblib'
    metadata_path = f'models/aqi_model_{horizon}_metadata.joblib'
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model not found for {horizon} forecast. Please train model first."
        )
    
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    # Get recent data (need historical data for lag features)
    with AQIFeatureStore() as fs:
        data = list(fs.raw_features.find({}).sort('timestamp', -1).limit(100))
    
    if len(data) < 50:
        raise HTTPException(
            status_code=503,
            detail="Insufficient data for prediction. Need at least 50 records."
        )
    
    # Prepare features
    df = pd.DataFrame(data).sort_values('timestamp').reset_index(drop=True)
    
    # Engineer all features (matches training pipeline)
    df = engineer_features(df)
    
    # Get features in correct order
    feature_cols = metadata['feature_cols']
    
    # Get latest row (most recent data point)
    X = df[feature_cols].iloc[-1:].fillna(0)
    
    # Predict
    prediction = model.predict(X)[0]
    prediction = max(1, min(5, prediction))  # Clip to 1-5 scale
    
    # Calculate forecast time
    latest_time = df['timestamp'].iloc[-1]
    hours = int(horizon.replace('h', ''))
    forecast_time = latest_time + pd.Timedelta(hours=hours)
    
    return ForecastResponse(
        horizon=horizon,
        forecast_time=forecast_time,
        predicted_aqi=round(prediction, 2),
        aqi_category=get_aqi_category(prediction),
        model_name=metadata['model_name'],
        mae=metadata['metrics']['mae'],
        r2=metadata['metrics']['r2']
    )


@app.get("/forecast/all", response_model=List[ForecastResponse], tags=["Forecast"])
async def get_all_forecasts():
    """Get forecasts for all horizons (24h, 48h, 72h)"""
    forecasts = []
    
    for horizon in ['24h', '48h', '72h']:
        try:
            forecast = await get_forecast(horizon)
            forecasts.append(forecast)
        except HTTPException:
            # Skip if model not available
            continue
    
    if not forecasts:
        raise HTTPException(
            status_code=404,
            detail="No trained models available"
        )
    
    return forecasts


@app.get("/history", tags=["Data"])
async def get_history(days: int = 7):
    """
    Get historical AQI data
    
    Args:
        days: Number of days of history (default: 7)
    """
    with AQIFeatureStore() as fs:
        cutoff = datetime.now() - timedelta(days=days)
        data = list(fs.raw_features.find(
            {'timestamp': {'$gte': cutoff}}
        ).sort('timestamp', 1))
        
        if not data:
            raise HTTPException(status_code=404, detail="No historical data available")
        
        # Format response
        history = []
        for record in data:
            history.append({
                'timestamp': record['timestamp'],
                'aqi': record['aqi'],
                'aqi_category': get_aqi_category(record['aqi']),
                'pm2_5': record['pm2_5'],
                'pm10': record['pm10'],
                'temperature': record.get('temperature'),
                'humidity': record.get('humidity')
            })
        
        return {
            'location': 'Karachi',
            'days': days,
            'record_count': len(history),
            'data': history
        }


# Run with: uvicorn src.forecast_api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)