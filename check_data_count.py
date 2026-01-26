"""
Check MongoDB data count - for GitHub Actions verification
"""
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

mongodb_uri = os.getenv('MONGODB_URI')

if not mongodb_uri:
    print("❌ MONGODB_URI not found!")
    exit(1)

client = MongoClient(mongodb_uri)
db = client['aqi_feature_store']

print("="*60)
print("📊 MONGODB DATA SUMMARY")
print("="*60)

raw_count = db['raw_features'].count_documents({})
training_count = db['training_data'].count_documents({})

print(f"\n✅ Raw Features (Live Data): {raw_count} documents")
print(f"✅ Training Data: {training_count} documents")

if raw_count > 0:
    latest = db['raw_features'].find_one(sort=[('ingestion_timestamp', -1)])
    print(f"\n📄 Latest Live Data:")
    print(f"   Timestamp: {latest.get('timestamp')}")
    print(f"   AQI: {latest.get('aqi')} (EPA 0-500 scale)")
    print(f"   PM2.5: {latest.get('pm2_5')} μg/m³")
    print(f"   Location: {latest.get('location')}")
    print(f"   Source: {latest.get('source')}")

print("\n" + "="*60)
print("✅ Verification complete!")
print("="*60)

client.close()