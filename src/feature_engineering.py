"""
Feature Engineering for AQI Prediction
Creates lag features, rolling statistics, time-based features, and interactions
"""

import pandas as pd
import numpy as np
from datetime import datetime


def engineer_all_features(df: pd.DataFrame, horizons: list = None) -> pd.DataFrame:
    """
    Engineer all features for AQI prediction from raw data
    
    Args:
        df: DataFrame with raw data (must have 'timestamp' column)
        horizons: List of forecast horizons (e.g., ['24h', '48h', '72h'])
    
    Returns:
        DataFrame with all engineered features including lags, rolling stats, and interactions
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp (critical for time-series features)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # ===== TIME-BASED FEATURES =====
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # ===== LAG FEATURES =====
    # Create lag features for PM2.5, PM10, and AQI
    for pollutant in ['pm2_5', 'pm10', 'aqi']:
        if pollutant in df.columns:
            for lag in [1, 3, 6, 12, 24]:
                df[f'{pollutant}_lag_{lag}h'] = df[pollutant].shift(lag)
    
    # ===== ROLLING STATISTICS =====
    # 24-hour rolling windows for key pollutants
    if 'pm2_5' in df.columns:
        df['pm2_5_rolling_24h_mean'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
        df['pm2_5_rolling_24h_std'] = df['pm2_5'].rolling(window=24, min_periods=1).std().fillna(0)
    
    if 'pm10' in df.columns:
        df['pm10_rolling_24h_mean'] = df['pm10'].rolling(window=24, min_periods=1).mean()
    
    if 'aqi' in df.columns:
        df['aqi_rolling_24h_mean'] = df['aqi'].rolling(window=24, min_periods=1).mean()
    
    # ===== INTERACTION FEATURES =====
    # Temperature × Humidity
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    
    # Wind Speed × Pressure
    if 'wind_speed' in df.columns and 'pressure' in df.columns:
        df['wind_pressure_interaction'] = df['wind_speed'] * df['pressure']
    
    # PM2.5 / PM10 ratio
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
    
    # NO2 / O3 ratio
    if 'no2' in df.columns and 'o3' in df.columns:
        df['no2_o3_ratio'] = df['no2'] / (df['o3'] + 1e-6)
    
    # ===== CREATE TARGET VARIABLES (for training) =====
    # Create forecast targets based on horizons
    if horizons and 'aqi' in df.columns:
        for horizon in horizons:
            hours = int(horizon.replace('h', ''))
            # Target is AQI shifted backward (future values)
            df[f'aqi_{horizon}_ahead'] = df['aqi'].shift(-hours)
    
    return df