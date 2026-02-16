"""
Model Registry - Store models in MongoDB GridFS
Works with both .env and Streamlit secrets
"""
from datetime import datetime
from typing import Dict, Optional, List
from pymongo import MongoClient
from gridfs import GridFS
import os
import joblib
import io

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


class ModelRegistry:
    """Model registry with GridFS for model file storage"""
    
    def __init__(self):
        """Initialize connection to MongoDB"""
        # Use global MONGODB_URI or try environment
        self.mongo_uri = MONGODB_URI or os.getenv('MONGODB_URI')
        
        if not self.mongo_uri:
            raise ValueError("MONGODB_URI not found! Check Streamlit secrets or .env file")
        
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            
            self.db = self.client['aqi_feature_store']
            self.registry = self.db['model_registry']
            self.fs = GridFS(self.db)
            
            print("✅ Connected to Model Registry (MongoDB)")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_object,  # ← The actual trained model
        metrics: Dict[str, float],
        feature_cols: List[str],  # ← Feature column names
        metadata: Optional[Dict] = None,
        stage: str = 'staging'
    ) -> str:
        """
        Register model and store in MongoDB GridFS
        
        Args:
            model_name: Name (e.g., 'aqi_forecast_24h')
            version: Version (e.g., 'v20260216_1430')
            model_object: Trained scikit-learn model
            metrics: Performance metrics (MAE, RMSE, R²)
            feature_cols: List of feature column names
            metadata: Additional info
            stage: 'staging', 'production', or 'archived'
        
        Returns:
            Model ID
        """
        # Serialize model to bytes
        model_bytes = io.BytesIO()
        joblib.dump(model_object, model_bytes)
        model_bytes.seek(0)
        
        # Store in GridFS
        file_id = self.fs.put(
            model_bytes.read(),
            filename=f"{model_name}_{version}.joblib",
            model_name=model_name,
            version=version
        )
        
        # Store metadata in registry collection
        model_entry = {
            'model_name': model_name,
            'version': version,
            'file_id': file_id,  # Reference to GridFS file
            'feature_cols': feature_cols,
            'metrics': metrics,
            'metadata': metadata or {},
            'stage': stage,
            'registered_at': datetime.now(),
            'registered_by': 'training_pipeline',
            'status': 'active'
        }
        
        result = self.registry.insert_one(model_entry)
        
        print(f"✅ Registered: {model_name} {version} → {stage}")
        print(f"   R²: {metrics.get('r2', 'N/A'):.3f} | MAE: {metrics.get('mae', 'N/A'):.3f}")
        print(f"   Stored in MongoDB GridFS")
        
        return str(result.inserted_id)
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ):
        """
        Load model from MongoDB GridFS
        
        Args:
            model_name: Name of model
            version: Specific version (if None, gets latest)
            stage: Filter by stage
        
        Returns:
            Tuple of (model_object, metadata)
        """
        # Get metadata
        metadata = self.get_model(model_name, version, stage)
        
        if not metadata:
            return None, None
        
        # Load from GridFS
        file_id = metadata['file_id']
        model_bytes = self.fs.get(file_id).read()
        
        # Deserialize
        model = joblib.load(io.BytesIO(model_bytes))
        
        return model, metadata
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[Dict]:
        """Get model metadata (not the model file itself)"""
        query = {'model_name': model_name, 'status': 'active'}
        
        if version:
            query['version'] = version
        if stage:
            query['stage'] = stage
        
        model = self.registry.find_one(query, sort=[('registered_at', -1)])
        return model
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        stage: Optional[str] = None
    ) -> List[Dict]:
        """List all registered models"""
        query = {'status': 'active'}
        
        if model_name:
            query['model_name'] = model_name
        if stage:
            query['stage'] = stage
        
        models = list(self.registry.find(query, sort=[('registered_at', -1)]))
        return models
    
    def promote_to_production(self, model_name: str, version: str) -> bool:
        """Promote model to production"""
        # Archive current production
        self.registry.update_many(
            {'model_name': model_name, 'stage': 'production', 'status': 'active'},
            {'$set': {'stage': 'archived', 'archived_at': datetime.now()}}
        )
        
        # Promote new
        result = self.registry.update_one(
            {'model_name': model_name, 'version': version, 'status': 'active'},
            {'$set': {'stage': 'production', 'promoted_at': datetime.now()}}
        )
        
        if result.modified_count > 0:
            print(f"✅ Promoted {model_name} {version} → PRODUCTION")
            return True
        else:
            print(f"⚠️ Model not found: {model_name} {version}")
            return False
    
    def get_production_model(self, model_name: str):
        """Load production model from MongoDB"""
        return self.load_model(model_name, stage='production')
    
    def compare_models(
        self,
        model_name: str,
        metric: str = 'r2'
    ) -> List[Dict]:
        """Compare all versions of a model by metric"""
        models = self.list_models(model_name)
        
        reverse = (metric == 'r2')
        
        sorted_models = sorted(
            models,
            key=lambda m: m['metrics'].get(metric, float('inf') if not reverse else -float('inf')),
            reverse=reverse
        )
        
        print(f"\n📊 Model Comparison: {model_name} (sorted by {metric})")
        print("="*70)
        
        for i, model in enumerate(sorted_models[:5], 1):
            stage_icon = "🏆" if model['stage'] == 'production' else "📦"
            print(f"{i}. {stage_icon} {model['version']} ({model['stage']})")
            print(f"   MAE: {model['metrics'].get('mae', 'N/A'):.3f} | "
                  f"RMSE: {model['metrics'].get('rmse', 'N/A'):.3f} | "
                  f"R²: {model['metrics'].get('r2', 'N/A'):.3f}")
        
        return sorted_models
    
    def get_model_history(self, model_name: str) -> List[Dict]:
        """Get full history of a model including archived versions"""
        models = list(self.registry.find(
            {'model_name': model_name},
            sort=[('registered_at', -1)]
        ))
        
        print(f"\n📜 Model History: {model_name}")
        print("="*70)
        
        for model in models:
            status_icon = {
                'production': '🏆',
                'staging': '📦',
                'archived': '📁'
            }.get(model['stage'], '❓')
            
            print(f"{status_icon} {model['version']} - {model['stage']}")
            print(f"   Registered: {model['registered_at']}")
            print(f"   MAE: {model['metrics'].get('mae', 'N/A'):.3f}")
            print(f"   R²: {model['metrics'].get('r2', 'N/A'):.3f}")
        
        return models
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Test the Model Registry"""
    print("="*70)
    print("🧪 MODEL REGISTRY TEST")
    print("="*70)
    
    with ModelRegistry() as registry:
        # List all models
        print("\n📋 All Registered Models:")
        models = registry.list_models()
        
        if not models:
            print("   No models registered yet")
        else:
            for model in models:
                print(f"   • {model['model_name']} {model['version']} ({model['stage']})")
        
        print("\n✅ Model Registry is working!")


if __name__ == "__main__":
    main()