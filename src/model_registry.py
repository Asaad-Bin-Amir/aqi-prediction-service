"""
Model Registry - Track model versions, metrics, and lifecycle
Uses MongoDB for centralized model metadata storage
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import json
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()


class ModelRegistry:
    """Centralized model registry for tracking versions and performance"""
    
    def __init__(self):
        """Initialize connection to MongoDB model registry"""
        self.mongo_uri = os.getenv('MONGODB_URI')
        if not self.mongo_uri:
            raise ValueError("MONGODB_URI not found in environment!")
        
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client['aqi_feature_store']
        self.registry = self.db['model_registry']
        
        print("✅ Connected to Model Registry (MongoDB)")
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None,
        stage: str = 'staging'
    ) -> str:
        """
        Register a new model version
        
        Args:
            model_name: Name of model (e.g., 'aqi_forecast_24h')
            version: Version identifier (e.g., 'v20260209_0200')
            model_path: Path to saved model file
            metrics: Performance metrics (MAE, RMSE, R²)
            metadata: Additional info (algorithm, features, etc.)
            stage: 'staging', 'production', or 'archived'
        
        Returns:
            Model ID
        """
        model_entry = {
            'model_name': model_name,
            'version': version,
            'model_path': model_path,
            'metrics': metrics,
            'metadata': metadata or {},
            'stage': stage,
            'registered_at': datetime.now(),
            'registered_by': 'training_pipeline',
            'status': 'active'
        }
        
        result = self.registry.insert_one(model_entry)
        
        print(f"✅ Registered: {model_name} {version} → {stage}")
        print(f"   MAE: {metrics.get('mae', 'N/A'):.2f}")
        
        return str(result.inserted_id)
    
    def get_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Retrieve model metadata
        
        Args:
            model_name: Name of model
            version: Specific version (if None, gets latest)
            stage: Filter by stage (production/staging)
        
        Returns:
            Model metadata dict
        """
        query = {'model_name': model_name, 'status': 'active'}
        
        if version:
            query['version'] = version
        
        if stage:
            query['stage'] = stage
        
        # Get latest if no version specified
        model = self.registry.find_one(
            query,
            sort=[('registered_at', -1)]
        )
        
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
        
        models = list(self.registry.find(
            query,
            sort=[('registered_at', -1)]
        ))
        
        return models
    
    def promote_to_production(
        self,
        model_name: str,
        version: str
    ) -> bool:
        """
        Promote model to production
        Demotes current production model to staging
        
        Args:
            model_name: Name of model
            version: Version to promote
        
        Returns:
            Success status
        """
        # Demote current production model
        self.registry.update_many(
            {
                'model_name': model_name,
                'stage': 'production',
                'status': 'active'
            },
            {
                '$set': {
                    'stage': 'archived',
                    'archived_at': datetime.now()
                }
            }
        )
        
        # Promote new model
        result = self.registry.update_one(
            {
                'model_name': model_name,
                'version': version,
                'status': 'active'
            },
            {
                '$set': {
                    'stage': 'production',
                    'promoted_at': datetime.now()
                }
            }
        )
        
        if result.modified_count > 0:
            print(f"✅ Promoted {model_name} {version} → PRODUCTION")
            return True
        else:
            print(f"⚠️ Model not found: {model_name} {version}")
            return False
    
    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """Get current production model"""
        return self.get_model(model_name, stage='production')
    
    def compare_models(
        self,
        model_name: str,
        metric: str = 'mae'
    ) -> List[Dict]:
        """
        Compare all versions of a model by metric
        
        Args:
            model_name: Name of model
            metric: Metric to sort by (mae, rmse, r2)
        
        Returns:
            List of models sorted by metric
        """
        models = self.list_models(model_name)
        
        # Sort by metric (ascending for mae/rmse, descending for r2)
        reverse = (metric == 'r2')
        
        sorted_models = sorted(
            models,
            key=lambda m: m['metrics'].get(metric, float('inf')),
            reverse=reverse
        )
        
        print(f"\n📊 Model Comparison: {model_name} (sorted by {metric})")
        print("="*70)
        
        for i, model in enumerate(sorted_models[:5], 1):
            stage_icon = "🏆" if model['stage'] == 'production' else "📦"
            print(f"{i}. {stage_icon} {model['version']} ({model['stage']})")
            print(f"   MAE: {model['metrics'].get('mae', 'N/A'):.2f} | "
                  f"RMSE: {model['metrics'].get('rmse', 'N/A'):.2f} | "
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
            print(f"   MAE: {model['metrics'].get('mae', 'N/A'):.2f}")
        
        return models
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        print("✅ Closed Model Registry connection")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
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
        
        # Example: Register a test model
        print("\n📝 Example: Register Test Model")
        test_metrics = {'mae': 15.5, 'rmse': 19.2, 'r2': 0.82}
        
        registry.register_model(
            model_name='aqi_forecast_24h',
            version='v_test_001',
            model_path='models/test.pkl',
            metrics=test_metrics,
            metadata={'algorithm': 'XGBoost', 'features': 50},
            stage='staging'
        )
        
        print("\n✅ Model Registry is working!")


if __name__ == "__main__":
    main()