"""
AQI Feature Store - MongoDB connection for time-series data
Works with both .env and Streamlit secrets
"""
from pymongo import MongoClient
from datetime import datetime
import os

# Try Streamlit secrets first (for cloud deployment)
MONGODB_URI = None
try:
    import streamlit as st
    if hasattr(st, 'secrets') and 'MONGODB_URI' in st.secrets:
        MONGODB_URI = st.secrets["MONGODB_URI"]
except:
    pass

# Fallback to dotenv (for local development)
if not MONGODB_URI:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        MONGODB_URI = os.getenv('MONGODB_URI')
    except:
        pass


class AQIFeatureStore:
    """MongoDB Feature Store for AQI data"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.mongo_uri = MONGODB_URI or os.getenv('MONGODB_URI')
        
        if not self.mongo_uri:
            raise ValueError("MONGODB_URI not found! Check Streamlit secrets or .env file")
        
        self.client = None
        self.db = None
        self.raw_features = None
    
    def __enter__(self):
        """Context manager entry"""
        try:
            # FIX: Use self.mongo_uri (not self.mongodb_uri)
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.server_info()
            
            self.db = self.client['aqi_feature_store']
            self.raw_features = self.db['raw_features']
            
            print("✅ Connected to AQI Feature Store (MongoDB Atlas)")
        except Exception as e:
            print(f"❌ Failed to connect: {str(e)}")
            raise ConnectionError(f"MongoDB connection failed: {str(e)}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.client:
            self.client.close()
            print("✅ Closed feature store connection")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
    
    def store_features(self, data: dict):
        """Store raw features in MongoDB"""
        try:
            data['timestamp'] = datetime.now()
            result = self.raw_features.insert_one(data)
            return result.inserted_id
        except Exception as e:
            print(f"Error storing features: {e}")
            return None
    
    def get_latest_features(self, limit: int = 100):
        """Retrieve latest feature records"""
        try:
            cursor = self.raw_features.find().sort('timestamp', -1).limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"Error retrieving features: {e}")
            return []