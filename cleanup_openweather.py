import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))
from feature_store import AQIFeatureStore

with AQIFeatureStore() as fs:
    count = fs.raw_features.count_documents({'source': {'$regex': 'openweather'}})
    print(f"Found {count} OpenWeather records")
    if count > 0:
        result = fs.raw_features.delete_many({'source': {'$regex': 'openweather'}})
        print(f"✅ Deleted {result.deleted_count} records")