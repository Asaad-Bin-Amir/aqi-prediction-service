"""
Automated Model Training
Runs on schedule to retrain models with latest data
"""
from datetime import datetime
from src.training_pipeline import train_all_horizons
from src.feature_store import AQIFeatureStore


def check_data_quality():
    """Check if enough data for training"""
    print("\n" + "="*70)
    print("🔍 DATA QUALITY CHECK")
    print("="*70)
    
    with AQIFeatureStore() as fs:
        total_records = fs.raw_features.count_documents({})
        
        print(f"\n📊 Total records: {total_records}")
        
        if total_records < 168:  # 1 week
            print(f"⚠️ WARNING: Only {total_records} records available")
            print(f"   Recommended minimum: 168 (1 week of hourly data)")
            print(f"   Training may proceed but accuracy will be limited")
        
        # Check AQI variance
        data = list(fs.raw_features.find({}))
        if data:
            import pandas as pd
            df = pd.DataFrame(data)
            
            aqi_std = df['aqi'].std()
            aqi_min = df['aqi'].min()
            aqi_max = df['aqi'].max()
            unique_aqi = df['aqi'].nunique()
            
            print(f"\n📈 AQI Statistics (1-5 scale):")
            print(f"   Range: {aqi_min:.1f} - {aqi_max:.1f}")
            print(f"   Std Dev: {aqi_std:.2f}")
            print(f"   Unique values: {unique_aqi}")
            
            if aqi_std < 0.3:
                print(f"\n⚠️ WARNING: Low AQI variance (σ = {aqi_std:.2f})")
                print(f"   Model may have difficulty learning patterns")
                print(f"   Recommendation: Collect more diverse data")
            else:
                print(f"\n✅ Good AQI variance (σ = {aqi_std:.2f})")
        
        # Quality gates
        quality_passed = True
        
        if total_records < 100:
            print(f"\n❌ QUALITY GATE FAILED: Insufficient data (need 100+)")
            quality_passed = False
        
        if total_records >= 100 and aqi_std < 0.2:
            print(f"\n⚠️ QUALITY WARNING: Low variance, but proceeding")
        
        return quality_passed, total_records


def main():
    """Main automated training function"""
    print("\n" + "="*70)
    print("🤖 AUTOMATED MODEL TRAINING")
    print(f"⏰ Started at: {datetime.now()}")
    print("="*70)
    
    # Check data quality
    quality_ok, record_count = check_data_quality()
    
    if not quality_ok:
        print("\n❌ Training aborted - data quality check failed")
        print("   Please collect more data before training")
        return
    
    # Train models
    print(f"\n✅ Data quality check passed ({record_count} records)")
    print("🚀 Starting model training...\n")
    
    try:
        train_all_horizons()
        
        print("\n" + "="*70)
        print("✅ AUTOMATED TRAINING COMPLETED SUCCESSFULLY")
        print(f"⏰ Finished at: {datetime.now()}")
        print("="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TRAINING FAILED")
        print(f"Error: {e}")
        print("="*70)
        raise


if __name__ == "__main__":
    main()