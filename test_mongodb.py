from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

print("="*60)
print("🔗 Testing MongoDB Atlas Connection")
print("="*60)

# New credentials
username = "aqi_user"
password = "qwerty123"

# Connection string - NO SPACES in hostname
uri = "mongodb+srv://aqi_user:qwerty123@prediction.gblgaqq.mongodb.net/?appName=AQI-Prediction"

print(f"\n✅ Using new credentials")
print(f"   Username: {username}")
print(f"   Cluster:  prediction.gblgaqq.mongodb.net")

try:
    print("🔌 Connecting to MongoDB Atlas...")
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    
    # Test connection
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB Atlas!")
    
    # List databases
    databases = client.list_database_names()
    print(f"\n📚 Available databases: {databases}")
    
    # Create/access feature store database
    db = client['aqi_feature_store']
    print(f"✅ Accessed database: 'aqi_feature_store'")
    
    # Create a test collection
    collection = db['aqi_features']
    print(f"✅ Accessed collection: 'aqi_features'")
    
    # Insert test document
    test_doc = {"test":  "connection", "status": "success"}
    result = collection.insert_one(test_doc)
    print(f"✅ Inserted test document with ID: {result.inserted_id}")
    
    # Read it back
    found = collection.find_one({"test": "connection"})
    print(f"✅ Retrieved document:  {found}")
    
    # Clean up test document
    collection. delete_one({"test": "connection"})
    print(f"✅ Cleaned up test document")
    
    client.close()
    print("\n" + "="*60)
    print("🎉 MongoDB Atlas is ready to use as your feature store!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Connection failed!")
    print(f"Error:  {e}")
    print("\nTroubleshooting:")
    print("1. Check your IP is whitelisted in MongoDB Atlas")
    print("2. Verify username/password are correct")
    print("3. Check network/firewall settings")