"""
Model Training Pipeline
Trains models for AQI forecasting (1-5 scale)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import os

from feature_store import AQIFeatureStore
from feature_engineering import engineer_all_features


def prepare_training_data(horizon='24h'):
    """
    Prepare training data for specific forecast horizon
    
    Args:
        horizon: Forecast horizon ('24h', '48h', '72h')
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"\n📊 Preparing training data for {horizon} forecast...")
    
    with AQIFeatureStore() as fs:
        # Get all raw features
        data = list(fs.raw_features.find({}).sort('timestamp', 1))
        
        if len(data) < 50:
            print(f"⚠️ WARNING: Only {len(data)} records available!")
            print(f"   Recommended minimum: 168 (1 week)")
        
        df = pd.DataFrame(data)
        
        # Engineer features
        df = engineer_all_features(df, horizons=[horizon])
        
        # Define target and features
        target_col = f'aqi_{horizon}_ahead'
        
        # Exclude non-feature columns
        exclude_cols = ['_id', 'timestamp', 'location', 'source', 
                       'weather_main', 'weather_description',
                       'aqi_24h_ahead', 'aqi_48h_ahead', 'aqi_72h_ahead']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Prepare X and y
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Drop any remaining NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"\n✅ Dataset prepared:")
        print(f"   Samples: {len(X)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Target range: {y.min():.1f} - {y.max():.1f} (1-5 scale)")
        
        # Time-based split (last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"\n📊 Train/Test split (time-based):")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Test:  {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, feature_cols


def train_and_evaluate_models(X_train, X_test, y_train, y_test, horizon):
    """
    Train multiple models and select best
    
    Args:
        X_train, X_test, y_train, y_test: Training/test data
        horizon: Forecast horizon
    
    Returns:
        best_model, best_model_name, metrics
    """
    print(f"\n🤖 Training models for {horizon} forecast...")
    
    models = {
        'RANDOM_FOREST': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GRADIENT_BOOSTING': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBOOST': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'RIDGE': Ridge(alpha=1.0),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Clip predictions to 1-5 range
        y_pred = np.clip(y_pred, 1, 5)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"      MAE:  {mae:.4f}")
        print(f"      RMSE: {rmse:.4f}")
        print(f"      R²:   {r2:.4f}")
    
    # Select best model (lowest MAE)
    best_name = min(results.keys(), key=lambda k: results[k]['mae'])
    best_model = results[best_name]['model']
    
    print(f"\n✅ Best model: {best_name}")
    print(f"   MAE:  {results[best_name]['mae']:.4f}")
    print(f"   RMSE: {results[best_name]['rmse']:.4f}")
    print(f"   R²:   {results[best_name]['r2']:.4f}")
    
    return best_model, best_name, results[best_name]


def save_model(model, model_name, horizon, metrics, feature_cols):
    """Save trained model and metadata"""
    
    os.makedirs('models', exist_ok=True)
    
    model_path = f'models/aqi_model_{horizon}.joblib'
    metadata_path = f'models/aqi_model_{horizon}_metadata.joblib'
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'horizon': horizon,
        'metrics': metrics,
        'feature_cols': feature_cols,
        'trained_at': datetime.now(),
        'aqi_scale': '1-5 (OpenWeather)'
    }
    joblib.dump(metadata, metadata_path)
    
    print(f"\n💾 Model saved:")
    print(f"   {model_path}")
    print(f"   {metadata_path}")


def train_all_horizons():
    """Train models for all forecast horizons"""
    print("\n" + "="*70)
    print("🚀 AQI FORECASTING - MODEL TRAINING PIPELINE")
    print("="*70)
    
    horizons = ['24h', '48h', '72h']
    
    for horizon in horizons:
        print(f"\n{'='*70}")
        print(f"📅 FORECAST HORIZON: {horizon}")
        print(f"{'='*70}")
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_cols = prepare_training_data(horizon)
        
        # Train models
        best_model, best_name, metrics = train_and_evaluate_models(
            X_train, X_test, y_train, y_test, horizon
        )
        
        # Save model
        save_model(best_model, best_name, horizon, metrics, feature_cols)
    
    print(f"\n{'='*70}")
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print(f"{'='*70}")


if __name__ == "__main__":
    train_all_horizons()