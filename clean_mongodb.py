"""
Clean MongoDB - Remove data with wrong AQI scale
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

print("="*60)
print("🧹 CLEAN MONGODB - REMOVE WRONG AQI SCALE DATA")
print("="*60)

mongodb_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongodb_uri)
db = client['aqi_feature_store']
collection = db['training_data']

# Show current status
total_before = collection.count_documents({})
print(f"\n📊 Current documents: {total_before}")

if total_before == 0:
    print("\n✅ Database is already empty!")
    client.close()
    exit(0)

# Show breakdown by source
print(f"\n📋 Documents by source:")
sources = collection.aggregate([
    {'$group': {'_id': '$source', 'count': {'$sum': 1}}}
])

for source in sources:
    print(f"   {source['_id']}: {source['count']} documents")

# Ask for confirmation
print("\n" + "="*60)
print("⚠️  WARNING: This will DELETE data!")
print("="*60)
print("\nWhat do you want to delete?")
print("1. Delete ONLY OpenWeather data (1-5 scale)")
print("2. Delete ALL data (clean slate)")
print("3. Cancel (don't delete anything)")

choice = input("\nEnter choice (1/2/3): ").strip()

if choice == "1":
    # Delete only OpenWeather data
    result = collection.delete_many({
        'source': {'$in': ['openweathermap_live', 'openweather']}
    })
    print(f"\n🗑️  Deleted {result.deleted_count} OpenWeather documents")
    
elif choice == "2":
    # Delete ALL data
    confirm = input("\n⚠️  Delete ALL data? Type 'YES' to confirm: ").strip()
    if confirm == "YES":
        result = collection.delete_many({})
        print(f"\n🗑️  Deleted {result.deleted_count} documents")
    else:
        print("\n❌ Cancelled")
        
elif choice == "3":
    print("\n❌ Cancelled - no data deleted")
    
else:
    print("\n❌ Invalid choice - no data deleted")

# Show final status
total_after = collection.count_documents({})
print(f"\n✅ Remaining documents: {total_after}")

client.close()

print("\n" + "="*60)
print("CLEANUP COMPLETE")
print("="*60)