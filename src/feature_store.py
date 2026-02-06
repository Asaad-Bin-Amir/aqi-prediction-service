"""
Feature Engineering Pipeline
Creates lag features, rolling statistics, and interactions
Designed for AQI 1-5 scale prediction
"""
import pandas as pd
import numpy as np
from datetime import datetime


def create_lag_features(df, columns=['aqi', 'pm2_5', 'pm10'], lags=[1, 3, 6, 12, 24]):
    """
    Create lag features for time-series prediction
    
    Args:
        df: DataFrame with hourly data
        columns: Columns to create lags for
        lags: List of lag hours
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    
    print(f"✅ Created lag features: {len(columns)} columns × {len(lags)} lags = {len(columns) * len(lags)} features")
    
    return df


def create_rolling_features(df, columns=['aqi', 'pm2_5', 'pm10', 'temperature'], windows=[6, 12, 24]):
    """
    Create rolling window statistics
    
    Args:
        df: DataFrame with hourly data
        columns: Columns to create rolling stats for
        windows: Window sizes in hours
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_{window}h_mean'] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std (volatility)
                df[f'{col}_rolling_{window}h_std'] = df[col].rolling(window=window, min_periods=1).std()
                
                # Rolling min/max
                df[f'{col}_rolling_{window}h_min'] = df[col].rolling(window=window, min_periods=1).min()
                df[f'{col}_rolling_{window}h_max'] = df[col].rolling(window=window, min_periods=1).max()
    
    feature_count = len(columns) * len(windows) * 5  # mean, std, min, max, range
    print(f"✅ Created rolling features: {feature_count} features")
    
    return df


def create_time_features(df, timestamp_col='timestamp'):
    """
    Extract temporal features from timestamp
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
    
    Returns:
        DataFrame with time features added
    """
    df = df.copy()
    
    if timestamp_col in df.columns:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract features
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding (hour)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Rush hour indicators
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 10)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    
    print(f"✅ Created time features: 10 features")
    
    return df


def create_interaction_features(df):
    """
    Create interaction features between weather and pollutants
    
    Args:
        df: DataFrame with base features
    
    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()
    
    # Temperature × Humidity (affects PM2.5 behavior)
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    
    # Wind × Pressure (dispersion potential)
    if 'wind_speed' in df.columns and 'pressure' in df.columns:
        df['wind_pressure_interaction'] = df['wind_speed'] * df['pressure']
    
    # PM2.5 × Humidity (hygroscopic growth)
    if 'pm2_5' in df.columns and 'humidity' in df.columns:
        df['pm25_humidity_interaction'] = df['pm2_5'] * (df['humidity'] / 100)
    
    # Temperature × O3 (photochemical)
    if 'temperature' in df.columns and 'o3' in df.columns:
        df['temp_o3_interaction'] = df['temperature'] * df['o3']
    
    print(f"✅ Created interaction features: 4 features")
    
    return df


def create_forecast_targets(df, horizons=['24h', '48h', '72h']):
    """
    Create forecast target variables (1-5 AQI scale)
    
    Args:
        df: DataFrame with 'aqi' column (1-5 scale)
        horizons: List of forecast horizons
    
    Returns:
        DataFrame with target columns
    """
    df = df.copy()
    
    for horizon in horizons:
        hours = int(horizon.replace('h', ''))
        
        # Shift AQI backward (look ahead into future)
        df[f'aqi_{horizon}_ahead'] = df['aqi'].shift(-hours)
    
    print(f"\n✅ Created forecast targets for: {', '.join(horizons)}")
    print(f"   AQI scale: 1-5 (OpenWeather)")
    print(f"   Target range: {df['aqi'].min():.0f} - {df['aqi'].max():.0f}")
    
    return df


def engineer_all_features(df, horizons=['24h', '48h', '72h']):
    """
    Apply all feature engineering steps
    
    Args:
        df: Raw DataFrame from MongoDB
        horizons: Forecast horizons to create targets for
    
    Returns:
        DataFrame with all features engineered
    """
    print("\n" + "="*70)
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n📊 Input data: {len(df)} records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   AQI range: {df['aqi'].min():.0f} - {df['aqi'].max():.0f} (1-5 scale)")
    
    # Apply feature engineering
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_interaction_features(df)
    df = create_forecast_targets(df, horizons=horizons)
    
    # Drop rows with NaN in target variables
    original_len = len(df)
    df = df.dropna(subset=[f'aqi_{h}_ahead' for h in horizons])
    dropped = original_len - len(df)
    
    print(f"\n🧹 Dropped {dropped} rows with missing targets")
    print(f"   Final dataset: {len(df)} records")
    print(f"   Total features: {len(df.columns)} columns")
    
    return df


# Test the pipeline
if __name__ == "__main__":
    from feature_store import AQIFeatureStore
    
    print("Testing feature engineering pipeline...")
    
    with AQIFeatureStore() as fs:
        # Get raw data
        data = list(fs.raw_features.find({}).sort('timestamp', 1))
        
        if not data:
            print("❌ No data in feature store!")
        else:
            import pandas as pd
            df = pd.DataFrame(data)
            
            print(f"\n📊 Raw data: {len(df)} records")
            
            # Engineer features
            df_engineered = engineer_all_features(df)
            
            print(f"\n✅ Feature engineering complete!")
            print(f"\nSample features:")
            print(df_engineered.columns.tolist()[:20])