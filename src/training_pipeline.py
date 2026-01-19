"""
Training Pipeline for AQI Prediction Service
Trains models on historical data and saves to MongoDB Feature Store
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn. metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import joblib
import os
from pathlib import Path
from feature_store import AQIFeatureStore


class AQITrainingPipeline:  
    """Complete training pipeline for AQI prediction"""
    
    def __init__(self, data_path: str, model_version: str = "v1"):
        """
        Initialize training pipeline
        
        Args:  
            data_path:  Path to CSV file with historical data
            model_version:  Version identifier for this training run
        """
        self. data_path = data_path
        self.model_version = model_version
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        # Create models directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        print(f"✅ Training Pipeline initialized (version: {model_version})")
    
    def load_data(self) -> pd.DataFrame:
        """Load historical data from CSV"""
        print(f"\n📂 Loading data from {self.data_path}...")
        
        df = pd.read_csv(self.data_path)
        print(f"✅ Loaded {len(df)} records")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Columns: {list(df.columns)}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with engineered features
        """
        print("\n🔧 Engineering features...")
        
        df = df.copy()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt. hour
        df['day_of_week'] = df['timestamp']. dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for month (12-month cycle)
        df['month_sin'] = np. sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Sort by timestamp for lag features
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Lag features (previous hours)
        for lag in [1, 3, 6, 12, 24]:
            df[f'pm2_5_lag_{lag}h'] = df['pm2_5'].shift(lag)
            df[f'pm10_lag_{lag}h'] = df['pm10'].shift(lag)
            df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)
        
        # Rolling statistics (past 24 hours)
        df['pm2_5_rolling_24h_mean'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
        df['pm2_5_rolling_24h_std'] = df['pm2_5'].rolling(window=24, min_periods=1).std()
        df['pm10_rolling_24h_mean'] = df['pm10'].rolling(window=24, min_periods=1).mean()
        df['aqi_rolling_24h_mean'] = df['aqi'].rolling(window=24, min_periods=1).mean()
        
        # Weather interactions
        df['temp_humidity_interaction'] = df['temp'] * df['humidity']
        df['wind_pressure_interaction'] = df['wind_speed'] * df['pressure']
        
        # Pollutant ratios
        df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1e-6)
        df['no2_o3_ratio'] = df['no2'] / (df['o3'] + 1e-6)
        
        # Drop rows with NaN values (from lag features)
        original_len = len(df)
        df = df.dropna()
        print(f"   Created time-based, lag, and interaction features")
        print(f"   Dropped {original_len - len(df)} rows with NaN values")
        print(f"   Final dataset:  {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def prepare_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2):
        """
        Split data into train and test sets
        
        Args:  
            df: DataFrame with engineered features
            test_size:  Proportion of data for testing
            
        Returns:  
            X_train, X_test, y_train, y_test
        """
        print(f"\n✂️ Splitting data (test_size={test_size})...")
        
        # Features to exclude
        exclude_cols = ['timestamp', 'city', 'lat', 'lon', 'aqi']
        
        # Separate features and target
        X = df.drop(columns=exclude_cols, errors='ignore')
        y = df['aqi']
        
        # Train-test split (time-based)
        split_idx = int(len(df) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        print(f"   Features: {len(X_train.columns)}")
        
        # Scale features
        print("   Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd. DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        print("\n🤖 Training models...")
        
        # Define models
        models_config = {
            'linear_regression': LinearRegression(),
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
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Train each model
        for name, model in models_config.items():
            print(f"   Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            print(f"      ✅ {name} trained")
        
        print(f"\n✅ Trained {len(self.models)} models")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n📊 Evaluating models...")
        
        for name, model in self.models. items():
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            self.results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"\n   {name}:")
            print(f"      MAE:    {mae:.4f}")
            print(f"      RMSE:  {rmse:.4f}")
            print(f"      R²:   {r2:.4f}")
        
        # Find best model
        best_model_name = min(self.results, key=lambda x: self.results[x]['mae'])
        print(f"\n🏆 Best model: {best_model_name} (MAE: {self.results[best_model_name]['mae']:.4f})")
        
        return best_model_name
    
    def save_models(self, best_model_name: str):
        """Save trained models and scaler"""
        print("\n💾 Saving models...")
        
        # Save best model
        best_model_path = self.models_dir / f"aqi_model_{self.model_version}. pkl"
        joblib.dump(self.models[best_model_name], best_model_path)
        print(f"   ✅ Saved best model:  {best_model_path}")
        
        # Save scaler
        scaler_path = self.models_dir / f"scaler_{self.model_version}.pkl"
        joblib. dump(self.scaler, scaler_path)
        print(f"   ✅ Saved scaler: {scaler_path}")
        
        # Save all models (optional)
        for name, model in self.models. items():
            model_path = self.models_dir / f"{name}_{self.model_version}.pkl"
            joblib.dump(model, model_path)
        
        print(f"   ✅ Saved all {len(self.models)} models")
        
        return best_model_path, scaler_path
    
    def save_to_feature_store(self, X_train, X_test, y_train, y_test, 
                             best_model_name: str, feature_names: list):
        """Save training data and metadata to MongoDB Feature Store"""
        print("\n💾 Saving to Feature Store...")
        
        with AQIFeatureStore() as fs:
            # Save training data
            fs.save_training_data(X_train, y_train, split='train', model_version=self.model_version)
            fs.save_training_data(X_test, y_test, split='test', model_version=self.model_version)
            
            # Prepare metadata
            metadata = {
                'best_model': best_model_name,
                'training_date': datetime.now().isoformat(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'num_features': len(feature_names),
                'feature_names': feature_names,
                'metrics': {
                    model_name: {
                        'mae': float(results['mae']),
                        'rmse':  float(results['rmse']),
                        'r2': float(results['r2'])
                    }
                    for model_name, results in self.results. items()
                }
            }
            
            # Save metadata
            fs.save_model_metadata(metadata, model_version=self.model_version)
            
        print("   ✅ Saved to Feature Store")
    
    def run(self):
        """Run the complete training pipeline"""
        print("="*60)
        print("🚀 AQI PREDICTION - TRAINING PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Engineer features
        df_engineered = self.engineer_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(df_engineered)
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Evaluate models
        best_model_name = self.evaluate_models(X_test, y_test)
        
        # Save models
        model_path, scaler_path = self.save_models(best_model_name)
        
        # Save to feature store
        self.save_to_feature_store(
            X_train, X_test, y_train, y_test,
            best_model_name, list(X_train.columns)
        )
        
        # Final summary
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"Model version: {self.model_version}")
        print(f"Best model: {best_model_name}")
        print(f"MAE: {self.results[best_model_name]['mae']:.4f}")
        print(f"RMSE: {self.results[best_model_name]['rmse']:.4f}")
        print(f"R²: {self.results[best_model_name]['r2']:.4f}")
        print(f"Model saved: {model_path}")
        print(f"Scaler saved: {scaler_path}")
        print("="*60)
        
        return best_model_name, self.results[best_model_name]


def main():
    """Main function to run training pipeline"""
    
    # Configuration
    DATA_PATH = "data/karachi_aqi_20260119_074012.csv"
    MODEL_VERSION = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found:  {DATA_PATH}")
        print("\n💡 Please update DATA_PATH in the script to point to your CSV file")
        return
    
    # Run pipeline
    pipeline = AQITrainingPipeline(data_path=DATA_PATH, model_version=MODEL_VERSION)
    best_model, metrics = pipeline.run()


if __name__ == "__main__":
    main()