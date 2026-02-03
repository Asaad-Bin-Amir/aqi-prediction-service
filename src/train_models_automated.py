"""
Automated Model Training Pipeline for GitHub Actions
Wrapper around training_pipeline.py with data quality checks
"""
import sys
from datetime import datetime
from pathlib import Path
import json
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from model_registry import ModelRegistry
from feature_store import AQIFeatureStore
from training_pipeline import AQIForecastPipeline


def check_data_quality():
    """Check if we have enough quality data to retrain"""
    print("\n🔍 Checking data quality...")
    
    with AQIFeatureStore() as fs:
        # Count total records
        total_count = fs.raw_features.count_documents({})
        
        # Count by source
        hybrid_count = fs.raw_features.count_documents({'source': 'hybrid_aqicn_openweather'})
        aqicn_count = fs.raw_features.count_documents({'source': 'aqicn'})
        
        # Get sample to check AQI range
        sample = list(fs.raw_features.find({}).limit(1000))
        
        if not sample:
            print("❌ No data found in MongoDB!")
            return False, {}
    
    df_sample = pd.DataFrame(sample)
    
    # Requirements
    MIN_RECORDS = 100  # Need at least 100 records
    MIN_AQI_RANGE = 20  # Need at least 20 AQI point variety
    
    aqi_min = df_sample['aqi'].min()
    aqi_max = df_sample['aqi'].max()
    aqi_range = aqi_max - aqi_min
    
    print(f"\n📊 Data Quality Report:")
    print(f"   Total records: {total_count}")
    print(f"   Hybrid (AQICN+OpenWeather): {hybrid_count}")
    print(f"   AQICN only: {aqicn_count}")
    print(f"   AQI range: {aqi_min:.0f} - {aqi_max:.0f} (span: {aqi_range:.0f})")
    
    # Check quality
    if total_count < MIN_RECORDS:
        print(f"\n⚠️ Not enough data!")
        print(f"   Current: {total_count} records")
        print(f"   Required: {MIN_RECORDS} records")
        print(f"   💡 Wait for more data to accumulate")
        return False, {}
    
    if aqi_range < MIN_AQI_RANGE:
        print(f"\n⚠️ Insufficient AQI variety!")
        print(f"   Current range: {aqi_range:.0f}")
        print(f"   Required: {MIN_AQI_RANGE}")
        print(f"   💡 Wait for more diverse AQI conditions")
        return False, {}
    
    print(f"\n✅ Data quality check PASSED!")
    print(f"   ✓ Sufficient records ({total_count} >= {MIN_RECORDS})")
    print(f"   ✓ Good AQI variety ({aqi_range:.0f} >= {MIN_AQI_RANGE})")
    
    data_info = {
        'total_records': total_count,
        'hybrid_records': hybrid_count,
        'aqicn_records': aqicn_count,
        'aqi_min': float(aqi_min),
        'aqi_max': float(aqi_max),
        'aqi_range': float(aqi_range)
    }
    
    return True, data_info


def save_training_metadata(best_models, results, data_info):
    """Save training run metadata for tracking"""
    print("\n📝 Saving training metadata...")
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_quality': data_info,
        'best_models': best_models,
        'performance': {
            horizon: {
                'model': model_name,
                'mae': float(results[f'{model_name}_{horizon}']['mae']),
                'rmse': float(results[f'{model_name}_{horizon}']['rmse']),
                'r2': float(results[f'{model_name}_{horizon}']['r2'])
            }
            for horizon, model_name in best_models.items()
        }
    }
    
    # Save to JSON
    metadata_file = Path('models') / 'training_history.json'
    
    # Load existing history
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(metadata)
    
    with open(metadata_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Training metadata saved to {metadata_file}")
    
    return metadata_file

def register_models_to_registry(best_models, results, model_version):
    """Register trained models to Model Registry"""
    print("\n📦 Registering models to Model Registry...")
    
    with ModelRegistry() as registry:
        for horizon, model_name in best_models.items():
            model_key = f'{model_name}_{horizon}'
            
            # Model file path
            model_path = f'models/aqi_forecast_{horizon}_{model_version}.pkl'
            
            # Metrics
            metrics = {
                'mae': float(results[model_key]['mae']),
                'rmse': float(results[model_key]['rmse']),
                'r2': float(results[model_key]['r2'])
            }
            
            # Metadata
            metadata = {
                'algorithm': model_name,
                'horizon': horizon,
                'version': model_version,
                'training_date': datetime.now().isoformat()
            }
            
            # Register
            registry.register_model(
                model_name=f'aqi_forecast_{horizon}',
                version=model_version,
                model_path=model_path,
                metrics=metrics,
                metadata=metadata,
                stage='staging'  # Start in staging
            )
            
            # Auto-promote to production if MAE < 20
            if metrics['mae'] < 20:
                print(f"   🏆 Auto-promoting {horizon} (MAE: {metrics['mae']:.2f} < 20)")
                registry.promote_to_production(
                    model_name=f'aqi_forecast_{horizon}',
                    version=model_version
                )
    
    print("✅ Models registered to Model Registry")

def main():
    """Main execution"""
    print("="*70)
    print("🤖 AUTOMATED MODEL TRAINING PIPELINE")
    print("   GitHub Actions - Weekly Retraining")
    print("="*70)
    
    # Step 1: Check data quality
    quality_ok, data_info = check_data_quality()
    
    if not quality_ok:
        print("\n⏸️ Training SKIPPED - waiting for more/better data")
        print("   This is normal! The system will try again next week.")
        return 0  # Not an error, just not ready
    
    # Step 2: Run training pipeline
    print("\n" + "="*70)
    print("🚀 Starting Training Pipeline...")
    print("="*70)
    
    try:
        MODEL_VERSION = f"v{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        pipeline = AQIForecastPipeline(model_version=MODEL_VERSION)
        best_models, results = pipeline.run()
        
        # Step 3: Save metadata
        save_training_metadata(best_models, results, data_info)
        
        # Step 4: Summary
        print("\n" + "="*70)
        print("✅ AUTOMATED TRAINING COMPLETE!")
        print("="*70)
        
        print(f"\n📊 Summary:")
        print(f"   Version: {MODEL_VERSION}")
        print(f"   Data: {data_info['total_records']} records")
        print(f"   Hybrid (AQICN): {data_info['hybrid_records']} records")
        
        print(f"\n🏆 Best Models:")
        for horizon, model_name in best_models.items():
            model_key = f'{model_name}_{horizon}'
            mae = results[model_key]['mae']
            print(f"   {horizon}: {model_name.upper()} (MAE: {mae:.2f})")
        
        print(f"\n💾 Models saved to: models/")
        print(f"📜 History: models/training_history.json")
        register_models_to_registry(best_models, results, MODEL_VERSION)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())