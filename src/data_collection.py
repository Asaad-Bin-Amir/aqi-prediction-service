"""
Data Collection Pipeline - OpenWeather Only
Complete air quality and weather data from single free API
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import requests
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

from src.feature_store import AQIFeatureStore

load_dotenv()

# Configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
CITY = "Karachi"
LAT = 24.8607
LON = 67.0011


def get_openweather_data():
    """Get air quality and weather data from OpenWeather"""
    try:
        # Air Pollution API
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
        air_response = requests.get(air_url, timeout=10)
        air_data = air_response.json()
        
        # Weather API
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric"
        weather_response = requests.get(weather_url, timeout=10)
        weather_data = weather_response.json()
        
        # Extract data
        air_components = air_data['list'][0]
        
        combined = {
            # Air quality (1-5 scale)
            'aqi': air_components['main']['aqi'],
            
            # Pollutants (µg/m³)
            'pm2_5': air_components['components']['pm2_5'],
            'pm10': air_components['components']['pm10'],
            'co': air_components['components'].get('co'),
            'no': air_components['components'].get('no'),
            'no2': air_components['components'].get('no2'),
            'o3': air_components['components'].get('o3'),
            'so2': air_components['components'].get('so2'),
            
            # Weather
            'temperature': weather_data['main']['temp'],
            'feels_like': weather_data['main']['feels_like'],
            'temp_min': weather_data['main']['temp_min'],
            'temp_max': weather_data['main']['temp_max'],
            'humidity': weather_data['main']['humidity'],
            'pressure': weather_data['main']['pressure'],
            'wind_speed': weather_data['wind']['speed'],
            'wind_direction': weather_data['wind'].get('deg', 0),
            'wind_gust': weather_data['wind'].get('gust'),
            'clouds': weather_data['clouds']['all'],
            'visibility': weather_data.get('visibility'),
            'weather_main': weather_data['weather'][0]['main'],
            'weather_description': weather_data['weather'][0]['description'],
            
            # Metadata
            'timestamp': datetime.now(timezone.utc),
            'location': CITY,
            'source': 'openweather'
        }
        
        print(f"\n✅ OpenWeather Data Collected:")
        print(f"   AQI: {combined['aqi']}/5")
        print(f"   PM2.5: {combined['pm2_5']:.2f} µg/m³")
        print(f"   PM10: {combined['pm10']:.2f} µg/m³")
        print(f"   Temperature: {combined['temperature']:.1f}°C")
        print(f"   Humidity: {combined['humidity']}%")
        print(f"   Wind: {combined['wind_speed']:.1f} m/s")
        print(f"   Weather: {combined['weather_description']}")
        
        return combined
        
    except Exception as e:
        print(f"\n❌ OpenWeather API error: {e}")
        import traceback
        traceback.print_exc()
        return None


def collect_and_store():
    """Collect data and store in MongoDB"""
    print("\n" + "="*70)
    print(" AQI DATA COLLECTION PIPELINE")
    print(f" Location: {CITY}, Pakistan")
    print(f" Source: OpenWeather (Free Tier)")
    print("="*70)
    
    with AQIFeatureStore() as fs:
        # Get latest record for duplicate detection
        latest = fs.raw_features.find_one(sort=[('timestamp', -1)])
        
        # Collect data
        data = get_openweather_data()
        
        if not data:
            print("\n❌ Data collection failed!")
            return
        
        # ========== IMPROVED DUPLICATE DETECTION ========== 
        if latest:
            # Make both timestamps timezone-aware for comparison
            latest_ts = latest['timestamp']
            current_ts = data['timestamp']
            
            # If latest is naive, make it UTC-aware
            if latest_ts.tzinfo is None:
                latest_ts = latest_ts.replace(tzinfo=timezone.utc)
            
            # Calculate time difference
            time_diff = (current_ts - latest_ts).total_seconds() / 60
            
            # ALWAYS store if more than 45 minutes old
            if time_diff >= 45:
                print(f"\n✅ Storing data ({time_diff:.0f} min since last collection)")
            else:
                # Check if data actually changed
                aqi_same = latest.get('aqi', 0) == data['aqi']
                pm25_same = abs(latest.get('pm2_5', 0) - data['pm2_5']) < 0.5
                temp_same = abs(latest.get('temperature', 0) - data['temperature']) < 0.5
                
                # Only skip if ALL are identical AND less than 45 min
                if aqi_same and pm25_same and temp_same:
                    print(f"\n⏭️ SKIPPING - Data unchanged ({time_diff:.0f} min ago)")
                    print("   (OpenWeather API caching - normal behavior)")
                    return
                else:
                    print(f"\n✅ Storing data - values changed (gap: {time_diff:.0f} min)")
        else:
            print("\n✅ Storing first record")
        
        # Save to MongoDB
        result = fs.raw_features.insert_one(data)
        
        print("\n✅ DATA SAVED TO MONGODB")
        print(f"   Record ID: {result.inserted_id}")
        print(f"   AQI: {data['aqi']}/5")
        print(f"   PM2.5: {data['pm2_5']:.2f} µg/m³")
        print(f"   Timestamp: {data['timestamp']}")


if __name__ == "__main__":
    collect_and_store()