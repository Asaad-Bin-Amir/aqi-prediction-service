"""
Clean duplicate records from MongoDB
Keeps only unique PM2.5 + AQI combinations
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.feature_store import AQIFeatureStore
from datetime import datetime

def clean_duplicates():
    """Remove duplicate records, keep only unique AQI/PM2.5 readings"""
    print("="*70)
    print("🧹 CLEANING DUPLICATE RECORDS")
    print("="*70)
    
    with AQIFeatureStore() as fs:
        # Get all records sorted by _id (MongoDB insertion order)
        all_records = list(fs.raw_features.find({}).sort('_id', 1))
        
        print(f"\n📊 Total records before cleaning: {len(all_records)}")
        
        seen = set()
        duplicates = []
        unique_count = 0
        
        for record in all_records:
            # Create unique key from AQI + PM2.5 (rounded to 2 decimals)
            pm25 = round(record.get('pm2_5', 0), 2)
            aqi = record.get('aqi', 0)
            key = (aqi, pm25)
            
            if key in seen:
                # This is a duplicate!
                duplicates.append(record['_id'])
            else:
                seen.add(key)
                unique_count += 1
        
        print(f"\n🔍 Analysis:")
        print(f"   Unique records: {unique_count}")
        print(f"   Duplicate records: {len(duplicates)}")
        
        if duplicates:
            response = input(f"\n❓ Delete {len(duplicates)} duplicates? (yes/no): ")
            if response.lower() == 'yes':
                result = fs.raw_features.delete_many({'_id': {'$in': duplicates}})
                print(f"\n✅ Deleted {result.deleted_count} duplicate records")
            else:
                print("\n⏭️ Skipped deletion")
                return
        else:
            print("\n✅ No duplicates found!")
            return
        
        # Show final count
        final_count = fs.raw_features.count_documents({})
        print(f"\n📊 Total records after cleaning: {final_count}")
        
        # Show unique AQI values
        unique_records = list(fs.raw_features.find({}).sort('_id', 1))
        if unique_records:
            import pandas as pd
            df = pd.DataFrame(unique_records)
            
            print(f"\n📈 Data Quality After Cleaning:")
            print(f"   AQI range: {df['aqi'].min():.0f} - {df['aqi'].max():.0f}")
            print(f"   AQI std dev: {df['aqi'].std():.2f}")
            print(f"   Unique AQI values: {df['aqi'].nunique()}")
            
            if 'pm2_5' in df.columns:
                print(f"   PM2.5 range: {df['pm2_5'].min():.2f} - {df['pm2_5'].max():.2f}")
                print(f"   PM2.5 std dev: {df['pm2_5'].std():.2f}")
            
            # Show sample of unique records
            print(f"\n📋 Sample of unique records:")
            # Get timestamp field name (could be 'timestamp', 'created_at', etc.)
            time_field = None
            for field in ['timestamp', 'created_at', 'collected_at', 'time']:
                if field in df.columns:
                    time_field = field
                    break
            
            if time_field:
                sample = df[[time_field, 'aqi', 'pm2_5']].head(10)
                print(sample.to_string(index=False))
            else:
                sample = df[['aqi', 'pm2_5']].head(10)
                print(sample.to_string(index=False))

if __name__ == "__main__":
    clean_duplicates()