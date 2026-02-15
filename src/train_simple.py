"""
Simplified training pipeline for limited data
Works with as few as 50 records
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
import os

from feature_store import AQIFeatureStore

def create_simple_features(df, forecast_hours=24):
    """Create minimal features that work with limited data"""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    
    # Basic lag features (only short-term)
    for lag in [1, 3, 6]:
        df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)
        df[f'pm2_5_lag_{lag}h'] = df['pm2_5'].shift(lag)
    
    # Simple rolling means
    for window in [3, 6]:
        df[f'aqi_rolling_mean_{window}h'] = df['aqi'].rolling(window).mean()
        df[f'pm2_5_rolling_mean_{window}h'] = df['pm2_5'].rolling(window).mean()
    
    # Time features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Weather features (current)
    weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'clouds']
    for feat in weather_features:
        if feat not in df.columns:
            df[feat] = 0
    
    # Target: AQI N hours ahead
    df[f'target_aqi_{forecast_hours}h'] = df['aqi'].shift(-forecast_hours)
    
    # Drop rows with NaN
    df_clean = df.dropna()
    
    return df_clean


def train_simple_model(horizon='24h'):
    """Train model with minimal feature requirements"""
    
    print(f"\n{'='*70}")
    print(f"🚀 TRAINING {horizon} FORECAST MODEL (SIMPLE)")
    print(f"{'='*70}\n")
    
    # Load data
    print("📊 Loading data from MongoDB...")
    with AQIFeatureStore() as fs:
        data = list(fs.raw_features.find({}).sort('timestamp', 1))
    
    if len(data) < 20:
        print(f"❌ Not enough data! Need at least 20 records, have {len(data)}")
        return None
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} records")
    
    # Engineer features
    forecast_hours = int(horizon.replace('h', ''))
    print(f"\n🔧 Creating features for {forecast_hours}h forecast...")
    df_features = create_simple_features(df, forecast_hours)
    
    print(f"✅ Feature engineering complete:")
    print(f"   Valid samples: {len(df_features)}")
    
    if len(df_features) < 10:
        print(f"❌ Not enough valid samples after feature engineering!")
        print(f"   Need at least 10, have {len(df_features)}")
        print(f"   This usually means data gaps are too large.")
        return None
    
    # Prepare train/test
    feature_cols = [col for col in df_features.columns 
                    if col not in ['timestamp', 'location', '_id', f'target_aqi_{forecast_hours}h', 'aqi']]
    
    X = df_features[feature_cols]
    y = df_features[f'target_aqi_{forecast_hours}h']
    
    # Time-based split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print(f"   Features: {len(feature_cols)}")
    
    # Train models
    models = {
        'RANDOM_FOREST': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        'GRADIENT_BOOST': GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42),
        'RIDGE': Ridge(alpha=1.0)
    }
    
    print(f"\n🤖 Training models...")
    
    best_model = None
    best_name = None
    best_mae = float('inf')
    results = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 1, 5)  # Clip to valid AQI range
        
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
        'horizon': horizon
    }
    
    joblib.dump(metadata, metadata_path)
    
    print(f"\n💾 Model saved:")
    print(f"   {model_path}")
    print(f"   {metadata_path}")
    
    return best_model, best_name, results[best_name]


def main():
    """Train all forecast horizons"""
    
    print("\n" + "="*70)
    print("🌍 AQI FORECASTING - SIMPLIFIED TRAINING PIPELINE")
    print("="*70)
    
    horizons = ['24h', '48h', '72h']
    
    for horizon in horizons:
        try:
            result = train_simple_model(horizon)
            if result is None:
                print(f"\n⚠️ Skipping {horizon} - not enough data")
        except Exception as e:
            print(f"\n❌ Error training {horizon}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()