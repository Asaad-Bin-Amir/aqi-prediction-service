"""
Hybrid Data Collection: AQICN (AQI) + OpenWeather (Weather Only)
- AQI/PM2.5: AQICN (accurate, from real monitoring stations)
- Weather: OpenWeather (complete and reliable)
- PM10: Estimated from PM2.5 (low importance 4.78%)
"""
import requests
from datetime import datetime
import os
from dotenv import load_dotenv
from feature_store import AQIFeatureStore

load_dotenv()

# API Configuration
AQICN_TOKEN = os.getenv('AQICN_API_TOKEN')
OPENWEATHER_KEY = os.getenv('OPENWEATHERMAP_API_KEY')

# Karachi coordinates
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011


class HybridCollector:
    """Collect AQI from AQICN + Weather from OpenWeather"""
    
    def __init__(self):
        """Initialize collectors"""
        if not AQICN_TOKEN:
            raise ValueError("❌ AQICN_API_TOKEN not found!")
        if not OPENWEATHER_KEY:
            raise ValueError("❌ OPENWEATHERMAP_API_KEY not found!")
        
        self.aqicn_token = AQICN_TOKEN
        self.openweather_key = OPENWEATHER_KEY
        
        print("="*60)
        print("🌆 HYBRID Data Collection - KARACHI")
        print("   AQI/PM2.5: AQICN (accurate)")
        print("   Weather: OpenWeather (reliable)")
        print("="*60)
    
    def get_aqicn_aqi(self):
        """Get AQI and PM2.5 from AQICN"""
        url = "https://api.waqi.info/feed/karachi/"
        params = {'token': self.aqicn_token}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'ok':
                return None
            
            d = data['data']
            iaqi = d.get('iaqi', {})
            
            aqi = d['aqi']
            pm25_aqi = iaqi.get('pm25', {}).get('v', None)
            pm25 = self._aqi_to_pm25(pm25_aqi) if pm25_aqi else None
            pm10 = pm25 * 2.2 if pm25 else None
            
            print(f"\n📡 AQICN (Air Quality):")
            print(f"   ✅ AQI: {aqi} (EPA scale)")
            print(f"   ✅ PM2.5: {pm25:.2f} μg/m³" if pm25 else "   ⚠️ PM2.5: N/A")
            print(f"   📊 PM10: {pm10:.2f} μg/m³ (estimated)" if pm10 else "   ⚠️ PM10: N/A")
            
            return {'aqi': aqi, 'pm2_5': pm25, 'pm10': pm10}
        
        except Exception as e:
            print(f"❌ AQICN error: {e}")
            return None
    
    def get_openweather_weather(self):
        """Get ONLY weather data from OpenWeather (NOT AQI!)"""
        weather_url = "http://api.openweathermap.org/data/2.5/weather"
        air_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        
        params = {'lat': KARACHI_LAT, 'lon': KARACHI_LON, 'appid': self.openweather_key}
        
        try:
            weather_response = requests.get(weather_url, params=params, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            air_response = requests.get(air_url, params=params, timeout=10)
            air_response.raise_for_status()
            air_data = air_response.json()
            
            components = air_data['list'][0]['components']
            
            print(f"\n📡 OpenWeather (Weather & Pollutants):")
            print(f"   🌡️ Temperature: {weather_data['main']['temp'] - 273.15:.1f}°C")
            print(f"   💧 Humidity: {weather_data['main']['humidity']}%")
            print(f"   🌀 Pressure: {weather_data['main']['pressure']} hPa")
            print(f"   💨 Wind: {weather_data['wind']['speed']} m/s")
            
            return {
                'no2': components.get('no2', 0), 'o3': components.get('o3', 0),
                'so2': components.get('so2', 0), 'co': components.get('co', 0),
                'nh3': components.get('nh3', 0), 'no': components.get('no', 0),
                'temperature': weather_data['main']['temp'] - 273.15,
                'feels_like': weather_data['main']['feels_like'] - 273.15,
                'pressure': weather_data['main']['pressure'],
                'humidity': weather_data['main']['humidity'],
                'wind_speed': weather_data['wind']['speed'],
                'wind_deg': weather_data['wind'].get('deg', 0),
                'clouds': weather_data['clouds']['all'],
                'weather_main': weather_data['weather'][0]['main'],
                'weather_description': weather_data['weather'][0]['description']
            }
        except Exception as e:
            print(f"❌ OpenWeather error: {e}")
            return None
    
    def _aqi_to_pm25(self, aqi):
        """Convert PM2.5 AQI to concentration"""
        if aqi is None:
            return None
        breakpoints = [(0,50,0.0,12.0),(51,100,12.1,35.4),(101,150,35.5,55.4),
                      (151,200,55.5,150.4),(201,300,150.5,250.4),(301,500,250.5,500.4)]
        for aqi_low, aqi_high, conc_low, conc_high in breakpoints:
            if aqi_low <= aqi <= aqi_high:
                return ((aqi - aqi_low) / (aqi_high - aqi_low)) * (conc_high - conc_low) + conc_low
        return aqi
    
    def collect_and_save(self):
        """Collect from both sources and merge"""
        aqicn_data = self.get_aqicn_aqi()
        weather_data = self.get_openweather_weather()
        
        if not aqicn_data:
            print("❌ Failed to get AQI from AQICN")
            return False
        
        if not weather_data:
            print("⚠️ Failed to get weather from OpenWeather, using defaults")
            weather_data = {'no2':0,'o3':0,'so2':0,'co':0,'nh3':0,'no':0,'temperature':25,
                          'feels_like':25,'pressure':1013,'humidity':50,'wind_speed':0,
                          'wind_deg':0,'clouds':0,'weather_main':'Unknown',
                          'weather_description':'Weather unavailable'}
        
        merged = {
            'location': 'Karachi', 'latitude': KARACHI_LAT, 'longitude': KARACHI_LON,
            'timestamp': datetime.now(), 'aqi': aqicn_data['aqi'],
            'pm2_5': aqicn_data['pm2_5'] or 0, 'pm10': aqicn_data['pm10'] or 0,
            **weather_data
        }
        
        print(f"\n✅ MERGED DATA:")
        print(f"   AQI: {merged['aqi']} (AQICN)")
        print(f"   PM2.5: {merged['pm2_5']:.2f} μg/m³ (AQICN)")
        print(f"   PM10: {merged['pm10']:.2f} μg/m³ (estimated)")
        print(f"   Temperature: {merged['temperature']:.1f}°C (OpenWeather)")
        print(f"   Humidity: {merged['humidity']}% (OpenWeather)")
        
        with AQIFeatureStore() as fs:
            batch_id = f"hybrid_aqicn_openweather_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            merged['batch_id'] = batch_id
            merged['source'] = 'hybrid_aqicn_openweather'
            merged['collection_type'] = 'live'
            fs.raw_features.insert_one(merged)
            print(f"\n💾 Saved to MongoDB")
            print(f"   Source: hybrid_aqicn_openweather")
        
        return True


def main():
    try:
        collector = HybridCollector()
        print("\n🔄 Collecting hybrid data...")
        success = collector.collect_and_save()
        if success:
            print("\n✅ Data collection complete!")
            print("="*60)
        else:
            print("\n❌ Data collection failed!")
            return 1
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())