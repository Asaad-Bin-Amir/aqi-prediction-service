"""
Optimized AQI Training with Model Registry Integration
Stores models in MongoDB GridFS for cloud deployment
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import os

from src.feature_store import AQIFeatureStore
from src.model_registry import ModelRegistry


def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering with interaction terms"""
    
    print("\n🔧 Advanced Feature Engineering...")
    
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
    
    # Rate of change
    df['pm2_5_change_1h'] = df['pm2_5'].diff(1)
    df['pm2_5_change_3h'] = df['pm2_5'].diff(3)
    df['pm2_5_change_6h'] = df['pm2_5'].diff(6)
    
    # Interactions
    df['pm2_5_x_wind'] = df['pm2_5'] * df['wind_speed']
    df['pm2_5_x_humidity'] = df['pm2_5'] * df['humidity']
    df['pm2_5_x_temp'] = df['pm2_5'] * df['temperature']
    df['wind_x_humidity'] = df['wind_speed'] * df['humidity']
    df['temp_x_humidity'] = df['temperature'] * df['humidity']
    df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1)
    
    # Weather changes
    df['temp_change'] = df['temperature'].diff(1)
    df['wind_change'] = df['wind_speed'].diff(1)
    df['pressure_change'] = df['pressure'].diff(1)
    
    print(f"   ✅ Created {len([c for c in df.columns if c not in ['timestamp', '_id', 'source']])} features")
    
    return df


def train_optimized_model(horizon='24h', use_registry=True):
    """Train with hyperparameter tuning and registry integration"""
    
    print(f"\n{'='*70}")
    print(f"🚀 OPTIMIZED TRAINING - {horizon} FORECAST")
    print(f"{'='*70}")
    
    # Load data
    print("\n📊 Loading data...")
    with AQIFeatureStore() as fs:
        data = list(fs.raw_features.find({'source': 'openweather_historical'}).sort('timestamp', 1))
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} records")
    
    # Feature engineering
    df = engineer_advanced_features(df)
    
    # Target
    forecast_hours = int(horizon.replace('h', ''))
    df['target_aqi'] = df['aqi'].shift(-forecast_hours)
    
    # Clean
    df_clean = df.dropna(subset=['target_aqi'])
    
    # Features
    exclude_cols = ['timestamp', 'location', '_id', 'target_aqi', 'aqi', 
                   'source', 'gap', 'weather_main', 'weather_description',
                   'co', 'no', 'no2', 'o3', 'so2', 'nh3', 'feels_like', 'temp_min', 
                   'temp_max', 'wind_direction', 'wind_gust', 'visibility', 'wind_deg']
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols].fillna(0)
    y = df_clean['target_aqi']
    
    print(f"\n✅ Features: {len(feature_cols)}")
    print(f"✅ Samples: {len(X)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train models
    print(f"\n🔍 Training and tuning models...")
    
    models = {}
    
    # Gradient Boosting
    print(f"\n   1️⃣ Gradient Boosting...")
    gb_params = {
        'n_estimators': [200, 250],
        'max_depth': [7, 9],
        'learning_rate': [0.08, 0.1],
        'min_samples_split': [5, 10],
        'subsample': [0.9, 1.0]
    }
    
    gb = GradientBoostingRegressor(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    gb_grid.fit(X_train, y_train)
    
    models['GradientBoosting'] = gb_grid.best_estimator_
    
    # Random Forest
    print(f"\n   2️⃣ Random Forest...")
    rf_params = {
        'n_estimators': [150, 200],
        'max_depth': [12, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train, y_train)
    
    models['RandomForest'] = rf_grid.best_estimator_
    
    # Evaluate
    print(f"\n🎯 MODEL COMPARISON:")
    
    best_model = None
    best_name = None
    best_r2 = -float('inf')
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 1, 5)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        print(f"\n   {name}:")
        print(f"      MAE:  {mae:.3f}")
        print(f"      RMSE: {rmse:.3f}")
        print(f"      R²:   {r2:.3f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
    
    print(f"\n✅ BEST MODEL: {best_name}")
    
    # Save to local models/ (for local testing)
    os.makedirs('models', exist_ok=True)
    
    model_filename = f'aqi_model_{horizon}.joblib'
    metadata_filename = f'aqi_model_{horizon}_metadata.joblib'
    
    model_path = f'models/{model_filename}'
    metadata_path = f'models/{metadata_filename}'
    
    joblib.dump(best_model, model_path)
    
    metadata = {
        'model_name': best_name,
        'feature_cols': feature_cols,
        'metrics': results[best_name],
        'trained_on': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'horizon': horizon,
        'model_type': 'forecasting',
        'optimization': 'GridSearchCV'
    }
    
    joblib.dump(metadata, metadata_path)
    
    print(f"\n💾 Saved locally:")
    print(f"   {model_path}")
    
    # Register in Model Registry (MongoDB GridFS)
    if use_registry:
        try:
            print(f"\n📝 Registering in Model Registry (MongoDB)...")
            
            version = datetime.now().strftime('v%Y%m%d_%H%M')
            
            with ModelRegistry() as registry:
                # Upload model object to MongoDB GridFS
                registry.register_model(
                    model_name=f'aqi_forecast_{horizon}',
                    version=version,
                    model_object=best_model,  # ← Pass actual model object
                    metrics=results[best_name],
                    feature_cols=feature_cols,  # ← Pass feature columns
                    metadata={
                        'algorithm': best_name,
                        'features': len(feature_cols),
                        'training_samples': len(X_train),
                        'test_samples': len(X_test),
                        'optimization': 'GridSearchCV',
                        'horizon': horizon
                    },
                    stage='staging'
                )
                
                print(f"✅ Registered in MongoDB Model Registry")
                
        except Exception as e:
            print(f"⚠️ Registry registration failed: {e}")
            print(f"   Model still saved locally")
            import traceback
            traceback.print_exc()
    
    return best_model, metadata


def main():
    print("\n" + "="*70)
    print("🌍 AQI PREDICTION - OPTIMIZED TRAINING PIPELINE")
    print("="*70)
    print("\n🎯 Target: R² > 0.70 for all models")
    print("⚡ Using advanced features + hyperparameter tuning")
    print("📝 Registering models in MongoDB Model Registry (GridFS)")
    
    horizons = ['24h', '48h', '72h']
    all_results = {}
    
    for horizon in horizons:
        model, metadata = train_optimized_model(horizon, use_registry=True)
        all_results[horizon] = metadata['metrics']
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS SUMMARY")
    print("="*70)
    
    for horizon, results in all_results.items():
        print(f"\n{horizon} Model:")
        print(f"   MAE:  {results['mae']:.3f}")
        print(f"   RMSE: {results['rmse']:.3f}")
        print(f"   R²:   {results['r2']:.3f} {'✅' if results['r2'] > 0.65 else '⚠️'}")
    
    avg_r2 = sum(r['r2'] for r in all_results.values()) / len(all_results)
    print(f"\n📈 Average R² across all models: {avg_r2:.3f}")
    
    print("\n🎉 Models saved locally AND in MongoDB GridFS!")
    print("   → Streamlit Cloud will load from MongoDB")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()