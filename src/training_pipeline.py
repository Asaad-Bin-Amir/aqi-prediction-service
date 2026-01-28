"""
Training Pipeline for AQI FORECASTING (3-Day Ahead)
Trains MULTIPLE ML models and selects the best for each horizon
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.base import clone
import xgboost as xgb
import joblib
import json
from pathlib import Path
from feature_store import AQIFeatureStore


class AQIForecastPipeline:
    """Complete training pipeline for AQI forecasting (3-day ahead) with model comparison"""
    
    def __init__(self, model_version: str = "v1"):
        """
        Initialize forecasting pipeline
        
        Args:
            model_version: Version identifier for this training run
        """
        self.model_version = model_version
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_models = {}
        
        # Create models directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        print(f"✅ AQI Forecasting Pipeline initialized (version: {model_version})")
    
    def load_data_from_mongodb(self) -> pd.DataFrame:
        """Load all training data from MongoDB Feature Store"""
        print(f"\n📂 Loading data from MongoDB Feature Store...")
        
        with AQIFeatureStore() as fs:
            # Get all training data
            cursor = fs.training_data.find({'model_version': 'v1'})
            data = list(cursor)
            
            if not data:
                print("   No v1 data found, loading all training data...")
                cursor = fs.training_data.find({})
                data = list(cursor)
            
            if not data:
                raise ValueError("❌ No training data found in MongoDB!")
            
            df = pd.DataFrame(data)
            df = df.drop('_id', axis=1, errors='ignore')
            
            # Remove MongoDB metadata columns
            metadata_cols = ['batch_id', 'split', 'model_version', 'creation_timestamp', 
                           'source', 'ingestion_timestamp']
            df = df.drop(columns=metadata_cols, errors='ignore')
        
        print(f"✅ Loaded {len(df)} records from MongoDB")
        print(f"   Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        
        return df
    
    def create_forecast_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create forecast targets: AQI 24h, 48h, 72h ahead
        
        Args:
            df: DataFrame with historical data
            
        Returns:
            DataFrame with forecast targets
        """
        print("\n🎯 Creating 3-day forecast targets...")
        
        df = df.copy()
        
        # Identify target column
        if 'aqi' in df.columns:
            target_col = 'aqi'
        elif 'target' in df.columns:
            target_col = 'target'
            df['aqi'] = df['target']
        else:
            raise ValueError("❌ No target column found!")
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            print("   Sorted by timestamp")
        
        # Create future targets using shift
        df['aqi_24h_ahead'] = df['aqi'].shift(-24)  # AQI 24 hours later
        df['aqi_48h_ahead'] = df['aqi'].shift(-48)  # AQI 48 hours later
        df['aqi_72h_ahead'] = df['aqi'].shift(-72)  # AQI 72 hours later
        
        # Drop rows with NaN targets (last 72 hours have no future data)
        original_len = len(df)
        df = df.dropna(subset=['aqi_24h_ahead', 'aqi_48h_ahead', 'aqi_72h_ahead'])
        
        print(f"   ✅ Created 3 forecast targets")
        print(f"   Dropped {original_len - len(df)} rows (last 72 hours with no future data)")
        print(f"   Final dataset: {len(df)} records")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for forecasting
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with engineered features
        """
        print("\n🔧 Engineering features for forecasting...")
        
        df = df.copy()
        
        # Time-based features (if timestamp exists)
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features (past values help predict future)
        for lag in [1, 3, 6, 12, 24]:
            if 'pm2_5' in df.columns:
                df[f'pm2_5_lag_{lag}h'] = df['pm2_5'].shift(lag)
            if 'pm10' in df.columns:
                df[f'pm10_lag_{lag}h'] = df['pm10'].shift(lag)
            if 'aqi' in df.columns:
                df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)
        
        # Rolling statistics (trends over past 24 hours)
        if 'pm2_5' in df.columns:
            df['pm2_5_rolling_24h_mean'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
            df['pm2_5_rolling_24h_std'] = df['pm2_5'].rolling(window=24, min_periods=1).std()
            df['pm2_5_rolling_24h_max'] = df['pm2_5'].rolling(window=24, min_periods=1).max()
            df['pm2_5_rolling_24h_min'] = df['pm2_5'].rolling(window=24, min_periods=1).min()
        
        if 'pm10' in df.columns:
            df['pm10_rolling_24h_mean'] = df['pm10'].rolling(window=24, min_periods=1).mean()
        
        if 'aqi' in df.columns:
            df['aqi_rolling_24h_mean'] = df['aqi'].rolling(window=24, min_periods=1).mean()
            df['aqi_rolling_24h_std'] = df['aqi'].rolling(window=24, min_periods=1).std()
        
        # Weather interactions
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        if 'wind_speed' in df.columns and 'pressure' in df.columns:
            df['wind_pressure_interaction'] = df['wind_speed'] * df['pressure']
        
        # Pollutant ratios
        if 'pm2_5' in df.columns and 'pm10' in df.columns:
            df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
        
        if 'no2' in df.columns and 'o3' in df.columns:
            df['no2_o3_ratio'] = df['no2'] / (df['o3'] + 1e-6)
        
        # Drop rows with NaN from lag features
        original_len = len(df)
        df = df.dropna()
        
        print(f"   Created lag, rolling, and interaction features")
        print(f"   Dropped {original_len - len(df)} rows with NaN")
        print(f"   Final: {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def prepare_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Split data into train and test sets for 3 forecast horizons
        
        Args:
            df: DataFrame with engineered features
            test_size: Proportion for testing
            
        Returns:
            X_train, X_test, y_train_dict, y_test_dict
        """
        print(f"\n✂️ Splitting data (test_size={test_size})...")
        
        # Features to exclude
        exclude_cols = ['timestamp', 'city', 'location', 'lat', 'lon', 'latitude', 
                       'longitude', 'aqi', 'target', 
                       'aqi_24h_ahead', 'aqi_48h_ahead', 'aqi_72h_ahead']
        
        # Separate features and targets
        X = df.drop(columns=exclude_cols, errors='ignore')
        
        y_24h = df['aqi_24h_ahead']
        y_48h = df['aqi_48h_ahead']
        y_72h = df['aqi_72h_ahead']
        
        # Time-based split (more realistic for time series)
        split_idx = int(len(df) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        
        y_train_24h = y_24h.iloc[:split_idx]
        y_test_24h = y_24h.iloc[split_idx:]
        
        y_train_48h = y_48h.iloc[:split_idx]
        y_test_48h = y_48h.iloc[split_idx:]
        
        y_train_72h = y_72h.iloc[:split_idx]
        y_test_72h = y_72h.iloc[split_idx:]
        
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        print(f"   Features: {len(X_train.columns)}")
        
        # Scale features
        print("   Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        y_train_dict = {
            '24h': y_train_24h,
            '48h': y_train_48h,
            '72h': y_train_72h
        }
        
        y_test_dict = {
            '24h': y_test_24h,
            '48h': y_test_48h,
            '72h': y_test_72h
        }
        
        return X_train, X_test, y_train_dict, y_test_dict
    
    def train_forecast_models(self, X_train, y_train_dict):
        """Train MULTIPLE ML models for each forecast horizon and compare"""
        print("\n🤖 Training multiple ML models for comparison...")
        
        # Define model candidates
        model_configs = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=42)
        }
        
        horizons = ['24h', '48h', '72h']
        
        # Train each model for each horizon
        for horizon in horizons:
            print(f"\n   📅 Training models for {horizon} ahead forecast:")
            
            for model_name, model in model_configs.items():
                print(f"      • {model_name:20s}", end=' ')
                
                # Clone the model to avoid reusing fitted models
                model_copy = clone(model)
                
                # Train
                model_copy.fit(X_train, y_train_dict[horizon])
                
                # Store with naming: model_horizon (e.g., xgboost_24h)
                self.models[f'{model_name}_{horizon}'] = model_copy
                print("✅")
        
        print(f"\n✅ Trained {len(model_configs)} models × {len(horizons)} horizons = {len(self.models)} total models")
    
    def evaluate_forecast_models(self, X_test, y_test_dict):
        """Evaluate ALL models and select best for each horizon"""
        print("\n📊 Evaluating all models and selecting best...")
        
        horizons = ['24h', '48h', '72h']
        model_types = ['xgboost', 'random_forest', 'gradient_boosting', 
                       'linear_regression', 'ridge']
        
        for horizon in horizons:
            print(f"\n   📅 {horizon} ahead forecast:")
            print("   " + "="*70)
            
            best_mae = float('inf')
            best_model_name = None
            
            for model_type in model_types:
                model_key = f'{model_type}_{horizon}'
                model = self.models[model_key]
                y_true = y_test_dict[horizon]
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                
                # Store results
                self.results[model_key] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                # Display with color coding (best in each metric)
                print(f"      {model_type:20s} │ MAE: {mae:7.2f} │ RMSE: {rmse:7.2f} │ R²: {r2:7.4f}")
                
                # Track best model by MAE
                if mae < best_mae:
                    best_mae = mae
                    best_model_name = model_type
            
            # Store best model for this horizon
            self.best_models[horizon] = best_model_name
            print(f"\n      🏆 Best model: {best_model_name.upper()} (MAE: {best_mae:.2f})")
        
        return self.results
    
    def save_models(self):
        """Save BEST models for each horizon + all models for reference"""
        print("\n💾 Saving models...")
        
        # Save BEST model for each horizon (for production use)
        print("\n   🏆 Saving best models (for deployment):")
        for horizon, best_model_name in self.best_models.items():
            model_key = f'{best_model_name}_{horizon}'
            model_path = self.models_dir / f"aqi_forecast_{horizon}_{self.model_version}.pkl"
            
            joblib.dump(self.models[model_key], model_path)
            print(f"      ✅ {horizon}: {best_model_name} → {model_path.name}")
        
        # Save scaler
        scaler_path = self.models_dir / f"scaler_{self.model_version}.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"\n   ✅ Saved scaler: {scaler_path.name}")
        
        # Save model selection metadata (which model was chosen for each horizon)
        metadata_path = self.models_dir / f"model_selection_{self.model_version}.json"
        selection_metadata = {
            horizon: {
                'model': best_model_name,
                'mae': self.results[f'{best_model_name}_{horizon}']['mae'],
                'rmse': self.results[f'{best_model_name}_{horizon}']['rmse'],
                'r2': self.results[f'{best_model_name}_{horizon}']['r2']
            }
            for horizon, best_model_name in self.best_models.items()
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(selection_metadata, f, indent=2)
        print(f"   ✅ Saved model selection metadata: {metadata_path.name}")
        
        # Optionally save ALL models for comparison/analysis
        all_models_dir = self.models_dir / "all_models"
        all_models_dir.mkdir(exist_ok=True)
        
        print(f"\n   📦 Saving all models to {all_models_dir.name}/ (for analysis):")
        for model_key, model in self.models.items():
            model_path = all_models_dir / f"{model_key}_{self.model_version}.pkl"
            joblib.dump(model, model_path)
        print(f"      ✅ Saved {len(self.models)} models")
        
        return self.models_dir
    
    def save_to_feature_store(self, X_train, X_test, y_train_dict, y_test_dict, feature_names):
        """Save training data and metadata to MongoDB Feature Store"""
        print("\n💾 Saving to Feature Store...")
        
        with AQIFeatureStore() as fs:
            # Save training splits (only first horizon to avoid duplication)
            fs.save_training_data(X_train, y_train_dict['24h'], split='train', 
                                model_version=self.model_version)
            fs.save_training_data(X_test, y_test_dict['24h'], split='test', 
                                model_version=self.model_version)
            
            # Prepare comprehensive metadata
            metadata = {
                'model_type': 'multi_model_forecast_3day',
                'models_trained': ['xgboost', 'random_forest', 'gradient_boosting', 
                                  'linear_regression', 'ridge'],
                'horizons': ['24h', '48h', '72h'],
                'best_models': self.best_models,
                'training_date': datetime.now().isoformat(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'num_features': len(feature_names),
                'feature_names': feature_names,
                'data_source': 'mongodb_feature_store',
                'performance_summary': {
                    horizon: {
                        'best_model': best_model,
                        'mae': float(self.results[f'{best_model}_{horizon}']['mae']),
                        'rmse': float(self.results[f'{best_model}_{horizon}']['rmse']),
                        'r2': float(self.results[f'{best_model}_{horizon}']['r2'])
                    }
                    for horizon, best_model in self.best_models.items()
                },
                'all_models_performance': {
                    model_key: {
                        'mae': float(results['mae']),
                        'rmse': float(results['rmse']),
                        'r2': float(results['r2'])
                    }
                    for model_key, results in self.results.items()
                }
            }
            
            # Save metadata
            fs.save_model_metadata(metadata, model_version=self.model_version)
            
        print("   ✅ Saved to Feature Store")
    
    def run(self):
        """Run the complete forecasting training pipeline with model comparison"""
        print("="*70)
        print("🚀 AQI 3-DAY FORECASTING - TRAINING PIPELINE")
        print("   Multi-Model Comparison & Selection")
        print("="*70)
        
        # Load data from MongoDB
        df = self.load_data_from_mongodb()
        
        # Create forecast targets
        df = self.create_forecast_targets(df)
        
        # Engineer features
        df_engineered = self.engineer_features(df)
        
        # Split data
        X_train, X_test, y_train_dict, y_test_dict = self.prepare_train_test_split(df_engineered)
        
        # Train multiple models
        self.train_forecast_models(X_train, y_train_dict)
        
        # Evaluate and select best models
        self.evaluate_forecast_models(X_test, y_test_dict)
        
        # Save best models
        models_dir = self.save_models()
        
        # Save to feature store
        self.save_to_feature_store(
            X_train, X_test, y_train_dict, y_test_dict,
            list(X_train.columns)
        )
        
        # Final summary
        print("\n" + "="*70)
        print("✅ FORECAST TRAINING COMPLETE!")
        print("="*70)
        print(f"Model version: {self.model_version}")
        print(f"Models saved to: {models_dir}/")
        
        print("\n🏆 Best Models Selected:")
        print("="*70)
        for horizon in ['24h', '48h', '72h']:
            best_model = self.best_models[horizon]
            model_key = f'{best_model}_{horizon}'
            results = self.results[model_key]
            
            print(f"\n   {horizon} ahead:")
            print(f"      Model: {best_model.upper()}")
            print(f"      MAE:   {results['mae']:.4f}")
            print(f"      RMSE:  {results['rmse']:.4f}")
            print(f"      R²:    {results['r2']:.4f}")
        
        print("\n" + "="*70)
        
        return self.best_models, self.results


def main():
    """Main function to run forecast training pipeline"""
    
    # Configuration
    MODEL_VERSION = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    print("="*70)
    print("🎯 AQI 3-DAY FORECAST MODEL TRAINING")
    print("   Multi-Model Comparison (XGBoost, RF, GB, Linear, Ridge)")
    print("   Using MongoDB Feature Store")
    print("="*70)
    
    # Run pipeline
    pipeline = AQIForecastPipeline(model_version=MODEL_VERSION)
    best_models, results = pipeline.run()


if __name__ == "__main__":
    main()