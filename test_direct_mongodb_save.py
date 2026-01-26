"""
Test saving data directly to MongoDB
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime

load_dotenv()

print("="*60)
print("🧪 TEST: Direct MongoDB Save")
print("="*60)

mongodb_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongodb_uri)
db = client['aqi_database']
collection = db['raw_features']

# Count before
count_before = collection.count_documents({})
print(f"\n📊 Documents before: {count_before}")

# Insert test document
test_doc = {
    'timestamp': datetime.now(),
    'location': 'Test',
    'aqi': 123,
    'pm2_5': 45.6,
    'source': 'test'
}

print(f"\n💾 Inserting test document...")
result = collection.insert_one(test_doc)
print(f"✅ Inserted with ID: {result.inserted_id}")

# Count after
count_after = collection.count_documents({})
print(f"📊 Documents after: {count_after}")

if count_after > count_before:
    print(f"\n✅ SUCCESS! Document saved correctly!")
    
    # Show the document
    saved_doc = collection.find_one({'_id': result.inserted_id})
    print(f"\n📄 Saved document:")
    for key, value in saved_doc.items():
        print(f"   {key}: {value}")
    
    # Clean up test document
    collection.delete_one({'_id': result.inserted_id})
    print(f"\n🗑️  Test document deleted")
else:
    print(f"\n❌ FAILED! Document not saved!")

client.close()

print("\n" + "="*60)