"""
Check which AQI sources we're using and their scales
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

print("="*60)
print("🔍 AQI DATA SOURCE ANALYSIS")
print("="*60)

# Check MongoDB data
mongodb_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongodb_uri)
db = client['aqi_feature_store']
collection = db['training_data']

# Get total count
total_count = collection.count_documents({})
print(f"\n📊 Total documents in MongoDB: {total_count}")

if total_count == 0:
    print("\n❌ No data in MongoDB yet!")
    client.close()
    exit(0)

# Get a sample document and show ALL fields
sample = collection.find_one()

if sample:
    print("\n📄 Sample Document Structure:")
    print("="*60)
    for key, value in sample.items():
        if key != '_id':
            print(f"   {key}: {value}")
    print("="*60)
    
    # Check what fields exist
    print(f"\n📋 Available Fields:")
    all_keys = set()
    for doc in collection.find().limit(100):
        all_keys.update(doc.keys())
    
    for key in sorted(all_keys):
        if key != '_id':
            print(f"   - {key}")
    
    # Try to find AQI-related fields
    print(f"\n🔍 Looking for AQI/Target field...")
    
    aqi_field = None
    possible_aqi_fields = ['aqi', 'target', 'y', 'label', 'pm2_5', 'AQI']
    
    for field in possible_aqi_fields:
        if field in all_keys:
            print(f"   ✅ Found potential AQI field: '{field}'")
            aqi_field = field
            break
    
    if aqi_field:
        # Analyze the AQI field
        print(f"\n📈 Analyzing '{aqi_field}' field in all {total_count} documents...")
        
        pipeline = [
            {
                '$match': {aqi_field: {'$ne': None}}
            },
            {
                '$group': {
                    '_id': None,
                    'min_value': {'$min': f'${aqi_field}'},
                    'max_value': {'$max': f'${aqi_field}'},
                    'avg_value': {'$avg': f'${aqi_field}'},
                    'count': {'$sum': 1}
                }
            }
        ]
        
        result = list(collection.aggregate(pipeline))
        
        if result and result[0]['count'] > 0:
            stats = result[0]
            min_val = stats['min_value']
            max_val = stats['max_value']
            avg_val = stats['avg_value']
            count = stats['count']
            
            print(f"   Documents with {aqi_field}: {count}")
            print(f"   Min: {min_val}")
            print(f"   Max: {max_val}")
            print(f"   Avg: {avg_val:.2f}")
            
            # Determine scale
            print(f"\n🎯 Scale Detection:")
            if max_val <= 5 and min_val >= 1:
                print(f"   ⚠️  WARNING: Values are 1-5 (OpenWeather scale)")
                print(f"   → MUST CONVERT TO 0-500 EPA SCALE!")
            elif max_val <= 500 and min_val >= 0:
                print(f"   ✅ GOOD: Values appear to be on 0-500 EPA scale")
            else:
                print(f"   ❓ Unusual range: {min_val} to {max_val}")
                print(f"   → This might be NORMALIZED or TRANSFORMED data")
    else:
        print(f"   ⚠️  No obvious AQI field found!")
        print(f"   → This might be FEATURE-ENGINEERED data")
        print(f"   → Check if this is training data (X features) without target (y)")

# Check if there's a source field
print(f"\n🗂️  Checking Data Sources...")
source_count = collection.count_documents({'source': {'$exists': True}})
print(f"   Documents with 'source' field: {source_count}")

if source_count > 0:
    sources = collection.aggregate([
        {'$match': {'source': {'$exists': True}}},
        {'$group': {'_id': '$source', 'count': {'$sum': 1}}}
    ])
    for source in sources:
        print(f"      {source['_id']}: {source['count']} documents")

# Check timestamps
print(f"\n📅 Checking Timestamps...")
timestamp_fields = ['timestamp', 'created_at', 'date', 'datetime']
for ts_field in timestamp_fields:
    ts_count = collection.count_documents({ts_field: {'$exists': True}})
    if ts_count > 0:
        print(f"   Found '{ts_field}' in {ts_count} documents")
        oldest = collection.find_one({ts_field: {'$exists': True}}, sort=[(ts_field, 1)])
        newest = collection.find_one({ts_field: {'$exists': True}}, sort=[(ts_field, -1)])
        if oldest and newest:
            print(f"      Oldest: {oldest.get(ts_field)}")
            print(f"      Newest: {newest.get(ts_field)}")

print("\n" + "="*60)
print("💡 ANALYSIS:")

# Determine what kind of data this is
if 'pm2_5' in all_keys and aqi_field is None:
    print("   This appears to be FEATURE DATA (X) without target (y)")
    print("   → Likely from feature engineering pipeline")
    print("   → May have been normalized/scaled")
elif aqi_field:
    print(f"   This appears to be COMPLETE DATA with target: {aqi_field}")
else:
    print("   Unable to determine data type")

print("="*60)

client.close()