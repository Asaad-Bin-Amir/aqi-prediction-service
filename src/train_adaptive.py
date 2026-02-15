"""
Adaptive Training Pipeline
Works with limited/gappy data, improves as data quality improves
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import os

from src.feature_store import AQIFeatureStore


def analyze_data_quality(df):
    """Analyze data to determine best training approach"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Check for consecutive sequences
    df['gap'] = df['timestamp'].diff()
    max_consecutive = 1
    current_seq = 1
    
    for i in range(1, len(df)):
        if df.loc[i, 'gap'] <= pd.Timedelta(hours=1.5):
            current_seq += 1
            max_consecutive = max(max_consecutive, current_seq)
        else:
            current_seq = 1
    
    return {
        'total_records': len(df),
        'max_consecutive': max_consecutive,
        'can_use_lags': max_consecutive >= 12,
        'can_forecast': max_consecutive >= 24
    }


def create_features(df, use_lags=False):
    """Create features based on data quality"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n🔧 Feature Engineering Mode: {'With Lags' if use_lags else 'Current Conditions Only'}")
    
    # Time features (always available)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Current weather features
    weather_features = ['pm2_5', 'pm10', 'temperature', 'humidity', 
                       'pressure', 'wind_speed', 'clouds']
    
    for feat in weather_features:
        if feat not in df.columns:
            df[feat] = 0
    
    # Lag features (only if data quality allows)
    if use_lags:
        for lag in [1, 3, 6]:
            df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)
            df[f'pm2_5_lag_{lag}h'] = df['pm2_5'].shift(lag)
        
        # Rolling features
        for window in [3, 6]:
            df[f'aqi_rolling_mean_{window}h'] = df['aqi'].rolling(window).mean()
            df[f'pm2_5_rolling_mean_{window}h'] = df['pm2_5'].rolling(window).mean()
    
    return df


def train_model(horizon='24h'):
    """Train model for given horizon"""
    
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING {horizon} FORECAST MODEL")
    print(f"{'='*70}")
    
    # Load data
    print("\n📊 Loading data from MongoDB...")
    with AQIFeatureStore() as fs:
        data = list(fs.raw_features.find({}).sort('timestamp', 1))
    
    if len(data) < 30:
        print(f"❌ Not enough data! Need at least 30 records, have {len(data)}")
        return None
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} records")
    
    # Analyze data quality
    quality = analyze_data_quality(df)
    print(f"\n📈 Data Quality Analysis:")
    print(f"   Total records: {quality['total_records']}")
    print(f"   Longest consecutive sequence: {quality['max_consecutive']}")
    print(f"   Can use lag features: {'Yes' if quality['can_use_lags'] else 'No'}")
    print(f"   Can do true forecasting: {'Yes' if quality['can_forecast'] else 'No'}")
    
    # Create features
    df_features = create_features(df, use_lags=quality['can_use_lags'])
    
    # For now, predict current AQI (nowcasting)
    # Will automatically become forecasting when we have enough consecutive data
    forecast_hours = int(horizon.replace('h', ''))
    if quality['can_forecast']:
        df_features[f'target_aqi'] = df_features['aqi'].shift(-forecast_hours)
        model_type = "Forecasting"
    else:
        df_features[f'target_aqi'] = df_features['aqi']  # Nowcast
        model_type = "Nowcasting"
    
    # Drop NaN
    df_clean = df_features.dropna()
    
    print(f"\n✅ Feature engineering complete:")
    print(f"   Valid samples: {len(df_clean)}")
    print(f"   Model type: {model_type}")
    
    if len(df_clean) < 20:
        print(f"❌ Not enough valid samples after feature engineering!")
        return None
    
    # Prepare features and target
    exclude_cols = ['timestamp', 'location', '_id', 'target_aqi', 'aqi', 
                   'source', 'gap', 'weather_main', 'weather_description',
                   'co', 'no', 'no2', 'o3', 'so2', 'feels_like', 'temp_min', 
                   'temp_max', 'wind_direction', 'wind_gust', 'visibility']
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean['target_aqi']
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print(f"   Features: {len(feature_cols)}")
    
    # Train multiple models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Ridge': Ridge(alpha=1.0)
    }
    
    print(f"\n🤖 Training models...")
    
    best_model = None
    best_name = None
    best_mae = float('inf')
    results = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 1, 5)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        print(f"      MAE:  {mae:.3f}")
        print(f"      RMSE: {rmse:.3f}")
        print(f"      R²:   {r2:.3f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name
    
    print(f"\n✅ Best Model: {best_name}")
    print(f"   MAE: {results[best_name]['mae']:.3f}")
    print(f"   R²:  {results[best_name]['r2']:.3f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    
    model_path = f'models/aqi_model_{horizon}.joblib'
    metadata_path = f'models/aqi_model_{horizon}_metadata.joblib'
    
    joblib.dump(best_model, model_path)
    
    metadata = {
        'model_name': best_name,
        'feature_cols': feature_cols,
        'metrics': results[best_name],
        'trained_on': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'horizon': horizon,
        'model_type': model_type,
        'data_quality': quality,
        'uses_lags': quality['can_use_lags']
    }
    
    joblib.dump(metadata, metadata_path)
    
    print(f"\n💾 Model saved:")
    print(f"   {model_path}")
    print(f"   {metadata_path}")
    
    return best_model, best_name, results[best_name]


def main():
    """Train all forecast horizons"""
    
    print("\n" + "="*70)
    print("🌍 AQI PREDICTION - ADAPTIVE TRAINING PIPELINE")
    print("="*70)
    
    horizons = ['24h', '48h', '72h']
    
    trained_count = 0
    for horizon in horizons:
        try:
            result = train_model(horizon)
            if result:
                trained_count += 1
            else:
                print(f"\n⚠️ Skipping {horizon} - insufficient data")
        except Exception as e:
            print(f"\n❌ Error training {horizon}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    if trained_count > 0:
        print(f"✅ TRAINING COMPLETE! ({trained_count}/{len(horizons)} models trained)")
    else:
        print("❌ NO MODELS TRAINED - Need more data")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()