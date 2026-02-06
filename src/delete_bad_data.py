"""
Delete broken synthetic data from MongoDB
"""
from feature_store import AQIFeatureStore

print("⚠️  WARNING: This will delete all training data from MongoDB!")
response = input("Are you sure? (yes/no): ")

if response.lower() == 'yes':
    with AQIFeatureStore() as fs:
        result = fs.training_data.delete_many({})
        print(f"✅ Deleted {result.deleted_count} documents")
else:
    print("❌ Cancelled")