"""
Verify raw_features collection has correct AQI scale
"""
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client['aqi_feature_store']
collection = db['raw_features']

print("="*60)
print("🔍 RAW FEATURES VERIFICATION")
print("="*60)

count = collection.count_documents({})
print(f"\n📊 Total raw features: {count}")

if count > 0:
    # Get latest document
    latest = collection.find_one(sort=[('ingestion_timestamp', -1)])
    
    print(f"\n📄 Latest Raw Feature:")
    print(f"   Timestamp: {latest.get('timestamp', 'N/A')}")
    print(f"   Location: {latest.get('location', 'N/A')}")
    print(f"   AQI: {latest.get('aqi', 'N/A')}")
    print(f"   PM2.5: {latest.get('pm2_5', 'N/A')}")
    print(f"   Source: {latest.get('source', 'N/A')}")
    print(f"   Batch ID: {latest.get('batch_id', 'N/A')}")
    
    aqi = latest.get('aqi', 0)
    
    print(f"\n🎯 AQI Scale Check:")
    if 1 <= aqi <= 5:
        print(f"   ❌ WRONG! AQI = {aqi} (1-5 scale)")
        print(f"   → You forgot to convert!")
    elif 6 <= aqi <= 500:
        print(f"   ✅ CORRECT! AQI = {aqi} (0-500 EPA scale)")
    else:
        print(f"   ❓ Unusual: AQI = {aqi}")
    
    # Check all documents
    all_aqi = [doc.get('aqi', 0) for doc in collection.find({}, {'aqi': 1})]
    print(f"\n📈 All {count} documents:")
    print(f"   Min AQI: {min(all_aqi)}")
    print(f"   Max AQI: {max(all_aqi)}")
    print(f"   Avg AQI: {sum(all_aqi)/len(all_aqi):.2f}")
    
else:
    print("\n✅ No raw features yet - collection is empty")
    print("   Run: python src/data_collection.py")

client.close()

print("\n" + "="*60)