"""
Feature Store - MongoDB Interface
Manages raw features and training data collections
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


class AQIFeatureStore:
    """
    MongoDB-based feature store for AQI prediction data
    
    Collections:
    - raw_features: Hourly air quality and weather data from APIs
    - training_data: Processed features for model training (cached)
    """
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.mongodb_uri = os.getenv('MONGODB_URI')
        
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI not found in environment variables")
        
        self.client = None
        self.db = None
        self.raw_features = None
        self.training_data = None
    
    def __enter__(self):
        """Context manager entry - connect to MongoDB"""
        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client['aqi_prediction']
        self.raw_features = self.db['raw_features']
        self.training_data = self.db['training_data']
        
        print("✅ Connected to AQI Feature Store (MongoDB Atlas)")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close MongoDB connection"""
        if self.client:
            self.client.close()
            print("✅ Closed feature store connection")
        
        return False
    
    def get_latest_features(self, limit=100):
        """
        Get latest raw features
        
        Args:
            limit: Number of recent records to return
        
        Returns:
            List of feature documents
        """
        return list(self.raw_features.find({}).sort('timestamp', -1).limit(limit))
    
    def get_feature_count(self):
        """Get total count of raw features"""
        return self.raw_features.count_documents({})
    
    def clear_all_data(self):
        """Clear all collections (use with caution!)"""
        self.raw_features.delete_many({})
        self.training_data.delete_many({})
        print("⚠️ All data cleared from feature store")


# Test the feature store
if __name__ == "__main__":
    print("Testing AQI Feature Store...\n")
    
    with AQIFeatureStore() as fs:
        count = fs.get_feature_count()
        print(f"📊 Total raw features: {count}")
        
        if count > 0:
            latest = fs.raw_features.find_one(sort=[('timestamp', -1)])
            print(f"\n🕐 Latest record:")
            print(f"   Timestamp: {latest.get('timestamp')}")
            print(f"   AQI: {latest.get('aqi')}")
            print(f"   PM2.5: {latest.get('pm2_5')}")
            print(f"   Location: {latest.get('location')}")
        else:
            print("\n⚠️ No data in feature store yet")
    
    print("\n✅ Feature store test complete!")