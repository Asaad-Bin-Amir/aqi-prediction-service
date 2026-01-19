"""
Utility Functions for AQI Prediction Service
Helper functions used across multiple modules
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import os


def calculate_aqi_from_pm25(pm25: float) -> int:
    """
    Calculate AQI from PM2.5 concentration using EPA breakpoints
    
    Args: 
        pm25: PM2.5 concentration in μg/m³
        
    Returns:
        AQI value (0-500)
    """
    # EPA AQI Breakpoints for PM2.5 (24-hour average)
    breakpoints = [
        (0.0, 12.0, 0, 50),       # Good
        (12.1, 35.4, 51, 100),    # Moderate
        (35.5, 55.4, 101, 150),   # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200),  # Unhealthy
        (150.5, 250.4, 201, 300), # Very Unhealthy
        (250.5, 350.4, 301, 400), # Hazardous
        (350.5, 500.4, 401, 500), # Hazardous
    ]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            # Linear interpolation formula
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
            return int(round(aqi))
    
    # If PM2.5 is beyond scale
    if pm25 > 500.4:
        return 500
    return 0


def calculate_aqi_from_pm10(pm10: float) -> int:
    """
    Calculate AQI from PM10 concentration using EPA breakpoints
    
    Args:
        pm10: PM10 concentration in μg/m³
        
    Returns:
        AQI value (0-500)
    """
    breakpoints = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm10 <= bp_hi:
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm10 - bp_lo) + aqi_lo
            return int(round(aqi))
    
    if pm10 > 604: 
        return 500
    return 0


def get_aqi_category(aqi: int) -> Dict[str, str]:
    """
    Get AQI category, color, and health message
    
    Args:
        aqi: AQI value
        
    Returns:
        Dictionary with category, color, and message
    """
    if aqi <= 50:
        return {
            "category": "Good",
            "color": "#00E400",
            "level": 1,
            "message": "Air quality is satisfactory, and air pollution poses little or no risk."
        }
    elif aqi <= 100:
        return {
            "category": "Moderate",
            "color": "#FFFF00",
            "level": 2,
            "message": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
        }
    elif aqi <= 150:
        return {
            "category": "Unhealthy for Sensitive Groups",
            "color": "#FF7E00",
            "level": 3,
            "message": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
        }
    elif aqi <= 200:
        return {
            "category": "Unhealthy",
            "color":  "#FF0000",
            "level": 4,
            "message": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
        }
    elif aqi <= 300:
        return {
            "category":  "Very Unhealthy",
            "color": "#8F3F97",
            "level": 5,
            "message": "Health alert: The risk of health effects is increased for everyone."
        }
    else: 
        return {
            "category":  "Hazardous",
            "color": "#7E0023",
            "level": 6,
            "message": "Health warning of emergency conditions:  everyone is more likely to be affected."
        }


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude coordinates
    
    Args:
        lat: Latitude (-90 to 90)
        lon: Longitude (-180 to 180)
        
    Returns:
        True if valid, False otherwise
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime for display
    
    Args:
        dt: datetime object
        
    Returns:
        Formatted string (YYYY-MM-DD HH: MM:SS)
    """
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse timestamp string to datetime
    
    Args:
        timestamp_str: Timestamp string
        
    Returns:
        datetime object
    """
    try:
        return pd.to_datetime(timestamp_str)
    except:
        return datetime.now()


def calculate_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive model evaluation metrics
    
    Args: 
        y_true: True values
        y_pred: Predicted values
        
    Returns: 
        Dictionary of metrics (MAE, RMSE, R², MAPE)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape)
    }


def get_feature_names() -> List[str]:
    """
    Get list of all feature names used in the model
    
    Returns:
        List of 42 feature names
    """
    # Base pollution and weather features
    base_features = [
        'pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2', 
        'temp', 'humidity', 'pressure', 'wind_speed'
    ]
    
    # Time-based features
    time_features = [
        'hour', 'day_of_week', 'day_of_month', 'month', 
        'is_weekend', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
    ]
    
    # Lag features (15 total)
    lag_features = []
    for pollutant in ['pm2_5', 'pm10', 'aqi']:
        for lag in [1, 3, 6, 12, 24]:
            lag_features.append(f'{pollutant}_lag_{lag}h')
    
    # Rolling statistics (4 total)
    rolling_features = [
        'pm2_5_rolling_24h_mean', 
        'pm2_5_rolling_24h_std',
        'pm10_rolling_24h_mean', 
        'aqi_rolling_24h_mean'
    ]
    
    # Interaction features (4 total)
    interaction_features = [
        'temp_humidity_interaction', 
        'wind_pressure_interaction',
        'pm2_5_pm10_ratio', 
        'no2_o3_ratio'
    ]
    
    all_features = (base_features + time_features + lag_features + 
                    rolling_features + interaction_features)
    
    return all_features


def load_latest_model(models_dir: str = "models") -> Tuple[object, object, str]:
    """
    Load the latest trained model and scaler
    
    Args: 
        models_dir: Directory containing model files
        
    Returns: 
        Tuple of (model, scaler, version_string)
        
    Raises:
        FileNotFoundError: If no models found
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Find latest model file
    model_files = list(models_path.glob("aqi_model_*.pkl"))
    
    if not model_files: 
        raise FileNotFoundError(
            f"No model files found in {models_dir}. "
            "Please run training_pipeline.py first."
        )
    
    # Get latest by modification time
    latest_model = max(model_files, key=os.path.getmtime)
    version = latest_model.stem.replace("aqi_model_", "")
    
    print(f"Loading model version: {version}")
    
    # Load model
    model = joblib.load(latest_model)
    
    # Load corresponding scaler
    scaler_file = models_path / f"scaler_{version}.pkl"
    
    if not scaler_file. exists():
        raise FileNotFoundError(f"Scaler file not found:  {scaler_file}")
    
    scaler = joblib. load(scaler_file)
    
    print(f"✅ Model and scaler loaded successfully")
    
    return model, scaler, version


def get_karachi_coordinates() -> Dict[str, float]:
    """
    Get Karachi city center coordinates
    
    Returns: 
        Dictionary with latitude, longitude, and city name
    """
    return {
        "lat": 24.8607,
        "lon": 67.0011,
        "city": "Karachi"
    }


def is_valid_aqi(aqi: float) -> bool:
    """
    Check if AQI value is within valid range
    
    Args: 
        aqi: AQI value to validate
        
    Returns:
        True if valid (0-500), False otherwise
    """
    return 0 <= aqi <= 500


def get_time_features(dt: datetime) -> Dict[str, float]:
    """
    Extract time-based features from datetime
    
    Args:
        dt: datetime object
        
    Returns:
        Dictionary with 9 time-based features
    """
    hour = dt.hour
    day_of_week = dt.weekday()  # 0 = Monday, 6 = Sunday
    day_of_month = dt.day
    month = dt.month
    
    return {
        'hour': hour,
        'day_of_week':  day_of_week,
        'day_of_month': day_of_month,
        'month': month,
        'is_weekend':  1 if day_of_week >= 5 else 0,
        'hour_sin':  np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np. pi * hour / 24),
        'month_sin': np. sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12)
    }


def calculate_interaction_features(data: Dict) -> Dict[str, float]:
    """
    Calculate interaction features from raw data
    
    Args: 
        data: Dictionary with raw features (temp, humidity, wind_speed, etc.)
        
    Returns:
        Dictionary with 4 interaction features
    """
    return {
        'temp_humidity_interaction': data. get('temp', 0) * data.get('humidity', 0),
        'wind_pressure_interaction': data. get('wind_speed', 0) * data.get('pressure', 1013),
        'pm2_5_pm10_ratio': data.get('pm2_5', 0) / (data.get('pm10', 1) + 1e-6),
        'no2_o3_ratio':  data.get('no2', 0) / (data.get('o3', 1) + 1e-6)
    }


def print_model_summary(model, metrics: Dict[str, float], version: str):
    """
    Print formatted model summary
    
    Args: 
        model:  Trained model object
        metrics: Dictionary of performance metrics
        version: Model version string
    """
    print("\n" + "="*60)
    print("🤖 MODEL SUMMARY")
    print("="*60)
    print(f"Model Type:     {type(model).__name__}")
    print(f"Version:       {version}")
    print(f"\n📊 Performance Metrics:")
    print(f"   MAE:        {metrics. get('mae', 0):.4f}")
    print(f"   RMSE:       {metrics.get('rmse', 0):.4f}")
    print(f"   R²:         {metrics.get('r2', 0):.4f}")
    print(f"   MAPE:       {metrics.get('mape', 0):.2f}%")
    print("="*60 + "\n")


def create_feature_dataframe(raw_data: Dict, historical_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create complete feature DataFrame from raw data
    
    Args:
        raw_data: Dictionary with current raw features
        historical_data:  Optional DataFrame with historical data for lag features
        
    Returns: 
        DataFrame with all 42 engineered features
    """
    # Start with raw data
    df = pd.DataFrame([raw_data])
    
    # Add time features
    if 'timestamp' in raw_data:
        dt = parse_timestamp(raw_data['timestamp'])
    else:
        dt = datetime. now()
    
    time_feats = get_time_features(dt)
    for key, value in time_feats. items():
        df[key] = value
    
    # Add interaction features
    interaction_feats = calculate_interaction_features(raw_data)
    for key, value in interaction_feats.items():
        df[key] = value
    
    # Add lag features (use zeros if no historical data)
    for pollutant in ['pm2_5', 'pm10', 'aqi']:
        for lag in [1, 3, 6, 12, 24]: 
            df[f'{pollutant}_lag_{lag}h'] = 0  # Default to 0
    
    # Add rolling features (use current values if no historical data)
    df['pm2_5_rolling_24h_mean'] = raw_data.get('pm2_5', 0)
    df['pm2_5_rolling_24h_std'] = 0
    df['pm10_rolling_24h_mean'] = raw_data.get('pm10', 0)
    df['aqi_rolling_24h_mean'] = raw_data.get('aqi', 0)
    
    return df


def validate_model_input(data: Dict) -> Tuple[bool, str]:
    """
    Validate input data for model prediction
    
    Args:
        data: Dictionary with input features
        
    Returns: 
        Tuple of (is_valid, error_message)
    """
    required_features = ['pm2_5', 'pm10', 'temp', 'humidity', 'pressure', 'wind_speed']
    
    for feature in required_features:
        if feature not in data:
            return False, f"Missing required feature: {feature}"
        
        if not isinstance(data[feature], (int, float)):
            return False, f"Invalid type for {feature}: expected number"
        
        if data[feature] < 0:
            return False, f"Invalid value for {feature}: must be non-negative"
    
    return True, ""


# Constants
KARACHI_COORDS = get_karachi_coordinates()
AQI_CATEGORIES = {
    "good": (0, 50),
    "moderate": (51, 100),
    "unhealthy_sensitive": (101, 150),
    "unhealthy": (151, 200),
    "very_unhealthy": (201, 300),
    "hazardous": (301, 500)
}