"""
Test MongoDB connection and check what's actually in there
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

print("="*60)
print("🔍 MONGODB CONNECTION TEST")
print("="*60)

# Get MongoDB URI
mongodb_uri = os.getenv('MONGODB_URI')

if not mongodb_uri:
    print("\n❌ MONGODB_URI not found in .env file!")
    exit(1)

print(f"\n✅ MongoDB URI found: {mongodb_uri[:30]}...{mongodb_uri[-10:]}")

# Connect
try:
    print("\n🔗 Connecting to MongoDB...")
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
    
    # Test connection
    client.server_info()
    print("✅ Connected successfully!")
    
    # List all databases
    print("\n📚 Available databases:")
    for db_name in client.list_database_names():
        print(f"   - {db_name}")
    
    # Check 'aqi_database'
    db = client['aqi_database']
    print(f"\n📊 Collections in 'aqi_database':")
    collections = db.list_collection_names()
    
    if not collections:
        print("   ⚠️  No collections found!")
    else:
        for coll_name in collections:
            count = db[coll_name].count_documents({})
            print(f"   - {coll_name}: {count} documents")
            
            # Show sample if exists
            if count > 0:
                sample = db[coll_name].find_one()
                print(f"      Sample keys: {list(sample.keys())[:10]}")
    
    # Check if data exists in OTHER databases
    print(f"\n🔍 Searching for AQI data in all databases...")
    found_data = False
    
    for db_name in client.list_database_names():
        if db_name in ['admin', 'local', 'config']:
            continue
            
        db = client[db_name]
        for coll_name in db.list_collection_names():
            count = db[coll_name].count_documents({})
            if count > 0:
                sample = db[coll_name].find_one()
                if 'aqi' in sample or 'pm2_5' in sample:
                    found_data = True
                    print(f"   ✅ Found AQI data: {db_name}.{coll_name} ({count} documents)")
    
    if not found_data:
        print("   ❌ No AQI data found in ANY database!")
    
    client.close()
    
except Exception as e:
    print(f"\n❌ MongoDB connection failed: {e}")

print("\n" + "="*60)