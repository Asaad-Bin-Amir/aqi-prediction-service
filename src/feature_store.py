"""
MongoDB-based Feature Store for AQI Prediction Service
Clean implementation - NO HopsWorks dependencies
"""
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import certifi

load_dotenv()


class AQIFeatureStore:
    """Feature store for AQI prediction using MongoDB Atlas"""

    def __init__(self):
        """Initialize connection to MongoDB Atlas"""
        self.uri = os.getenv('MONGODB_URI')
        if not self.uri:
            raise ValueError("❌ MONGODB_URI not found in .env file")

        # Connect with SSL certificate handling for GitHub Actions compatibility
        self.client = MongoClient(
            self.uri,
            serverSelectionTimeoutMS=5000,
            tlsCAFile=certifi.where()
        )
        self.db = self.client['aqi_feature_store']

        # Collections for different data types
        self.raw_features = self.db['raw_features']
        self.engineered_features = self.db['engineered_features']
        self.training_data = self.db['training_data']
        self.predictions = self.db['predictions']
        self.model_metadata = self.db['model_metadata']

        print("✅ Connected to AQI Feature Store (MongoDB Atlas)")

    def save_raw_features(self, data: pd.DataFrame, source: str = "api") -> str:
        """
        Save raw features from data collection

        Args:
            data: DataFrame with raw features
            source: Data source identifier (e.g., 'openweathermap_live', 'backfill')

        Returns:
            Batch ID for this insert
        """
        if data.empty:
            print("⚠️  Empty DataFrame, nothing to save")
            return None

        batch_id = f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        records = data.to_dict('records')
        for record in records:
            record['batch_id'] = batch_id
            record['source'] = source
            record['ingestion_timestamp'] = datetime.now()

            # Convert timestamp to datetime if it's a string
            if 'timestamp' in record and isinstance(record['timestamp'], str):
                record['timestamp'] = pd.to_datetime(record['timestamp'])

        result = self.raw_features.insert_many(records)
        print(f"✅ Saved {len(result.inserted_ids)} raw features (batch: {batch_id})")

        return batch_id

    def save_engineered_features(self, data: pd.DataFrame, feature_version: str = "v1") -> str:
        """
        Save engineered/transformed features

        Args:
            data: DataFrame with engineered features
            feature_version: Version identifier for features

        Returns:
            Batch ID for this insert
        """
        if data.empty:
            print("⚠️  Empty DataFrame, nothing to save")
            return None

        batch_id = f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        records = data.to_dict('records')
        for record in records:
            record['batch_id'] = batch_id
            record['feature_version'] = feature_version
            record['creation_timestamp'] = datetime.now()

        result = self.engineered_features.insert_many(records)
        print(f"✅ Saved {len(result.inserted_ids)} engineered features (batch: {batch_id})")

        return batch_id

    def save_training_data(self, X: pd.DataFrame, y: pd.Series,
                          split: str = "train", model_version: str = "v1") -> str:
        """
        Save training/validation/test data

        Args:
            X: Feature DataFrame
            y: Target Series
            split: 'train', 'val', or 'test'
            model_version: Model version identifier

        Returns:
            Batch ID for this insert
        """
        if X.empty:
            print("⚠️  Empty DataFrame, nothing to save")
            return None

        batch_id = f"{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        data = X.copy()
        data['target'] = y.values

        records = data.to_dict('records')
        for record in records:
            record['batch_id'] = batch_id
            record['split'] = split
            record['model_version'] = model_version
            record['creation_timestamp'] = datetime.now()

        result = self.training_data.insert_many(records)
        print(f"✅ Saved {len(result.inserted_ids)} {split} samples (batch: {batch_id})")

        return batch_id

    def save_predictions(self, data: pd.DataFrame, model_version: str = "v1") -> str:
        """
        Save model predictions

        Args:
            data: DataFrame with features and predictions
            model_version: Model version used for predictions

        Returns:
            Batch ID for this insert
        """
        if data.empty:
            print("⚠️  Empty DataFrame, nothing to save")
            return None

        batch_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        records = data.to_dict('records')
        for record in records:
            record['batch_id'] = batch_id
            record['model_version'] = model_version
            record['prediction_timestamp'] = datetime.now()

        result = self.predictions.insert_many(records)
        print(f"✅ Saved {len(result.inserted_ids)} predictions (batch: {batch_id})")

        return batch_id

    def save_model_metadata(self, metadata: dict, model_version: str = "v1") -> str:
        """
        Save model training metadata

        Args:
            metadata: Dictionary with model info (metrics, params, etc.)
            model_version: Model version identifier

        Returns:
            Insert ID
        """
        record = {
            'model_version': model_version,
            'timestamp': datetime.now(),
            **metadata
        }

        result = self.model_metadata.insert_one(record)
        print(f"✅ Saved model metadata (version: {model_version})")

        return str(result.inserted_id)

    def get_raw_features(self, batch_id: Optional[str] = None,
                        source: Optional[str] = None,
                        limit: int = 1000) -> pd.DataFrame:
        """
        Retrieve raw features

        Args:
            batch_id: Specific batch ID to retrieve
            source: Filter by source (e.g., 'openweathermap_live')
            limit: Maximum number of records to retrieve

        Returns:
            DataFrame with raw features
        """
        query = {}
        if batch_id:
            query['batch_id'] = batch_id
        if source:
            query['source'] = source

        cursor = self.raw_features.find(query).sort('timestamp', -1).limit(limit)
        data = list(cursor)

        if data:
            df = pd.DataFrame(data)
            df = df.drop('_id', axis=1, errors='ignore')
            return df
        return pd.DataFrame()

    def get_engineered_features(self, feature_version: str = "v1",
                               limit: int = 1000) -> pd.DataFrame:
        """Retrieve engineered features"""
        cursor = self.engineered_features.find(
            {'feature_version': feature_version}
        ).sort('creation_timestamp', -1).limit(limit)
        data = list(cursor)

        if data:
            df = pd.DataFrame(data)
            df = df.drop('_id', axis=1, errors='ignore')
            return df
        return pd.DataFrame()

    def get_training_data(self, split: str = "train",
                         model_version: str = "v1",
                         limit: Optional[int] = None) -> tuple:
        """
        Retrieve training data

        Args:
            split: 'train', 'val', or 'test'
            model_version: Model version identifier
            limit: Maximum number of records (None = all)

        Returns:
            (X, y) tuple of features and targets
        """
        query = {
            'split': split,
            'model_version': model_version
        }

        cursor = self.training_data.find(query)
        if limit:
            cursor = cursor.limit(limit)

        data = list(cursor)

        if data:
            df = pd.DataFrame(data)
            df = df.drop('_id', axis=1, errors='ignore')

            y = df['target']
            X = df.drop(['target', 'batch_id', 'split', 'model_version',
                        'creation_timestamp'], axis=1, errors='ignore')

            return X, y
        return pd.DataFrame(), pd.Series()

    def get_latest_predictions(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve latest predictions"""
        cursor = self.predictions.find().sort('prediction_timestamp', -1).limit(limit)
        data = list(cursor)

        if data:
            df = pd.DataFrame(data)
            df = df.drop('_id', axis=1, errors='ignore')
            return df
        return pd.DataFrame()

    def get_latest_raw_data(self, location: Optional[str] = None,
                           limit: int = 100) -> pd.DataFrame:
        """
        Get latest raw data for a location

        Args:
            location: City name (e.g., 'Karachi')
            limit: Number of records to retrieve

        Returns:
            DataFrame with latest raw data
        """
        query = {}
        if location:
            query['location'] = location

        cursor = self.raw_features.find(query).sort('timestamp', -1).limit(limit)
        data = list(cursor)

        if data:
            df = pd.DataFrame(data)
            df = df.drop('_id', axis=1, errors='ignore')
            return df
        return pd.DataFrame()

    def get_model_metadata(self, model_version: str = "v1") -> dict:
        """Retrieve model metadata"""
        result = self.model_metadata.find_one(
            {'model_version': model_version},
            sort=[('timestamp', -1)]
        )

        if result:
            result.pop('_id', None)
            return result
        return {}

    def get_stats(self) -> dict:
        """Get feature store statistics"""
        stats = {
            'raw_features_count': self.raw_features.count_documents({}),
            'engineered_features_count': self.engineered_features.count_documents({}),
            'training_data_count': self.training_data.count_documents({}),
            'predictions_count': self.predictions.count_documents({}),
            'model_metadata_count': self.model_metadata.count_documents({}),
        }

        # Get latest batch info
        latest = self.raw_features.find_one(sort=[('ingestion_timestamp', -1)])
        if latest:
            stats['latest_ingestion'] = latest.get('ingestion_timestamp')
            stats['latest_batch_id'] = latest.get('batch_id')
            stats['latest_source'] = latest.get('source')

        return stats

    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        print("✅ Closed feature store connection")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Test the feature store
if __name__ == "__main__":
    print("="*60)
    print("Testing AQI Feature Store")
    print("="*60)

    with AQIFeatureStore() as fs:
        # Get statistics
        print("\n📊 Feature Store Statistics:")
        stats = fs.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        # Get latest raw data
        print("\n📄 Latest Raw Data:")
        latest = fs.get_latest_raw_data(location='Karachi', limit=5)
        if not latest.empty:
            print(latest[['location', 'timestamp', 'aqi', 'pm2_5', 'temperature']].to_string())
        else:
            print("   No data found")

    print("\n✅ Feature store test complete!")