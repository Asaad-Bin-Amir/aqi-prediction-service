"""
Data Collection Module for AQI Prediction Service
Collects CURRENT/LIVE air quality data from OpenWeatherMap API and saves to MongoDB Feature Store
"""
import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from feature_store import AQIFeatureStore

load_dotenv()


class AQIDataCollector: 
    """Collects CURRENT AQI data from OpenWeatherMap API"""
    
    def __init__(self):
        """Initialize the data collector"""
        self.api_key = os.getenv('OPENWEATHERMAP_API_KEY')
        if not self.api_key:
            raise ValueError("❌ OPENWEATHERMAP_API_KEY not found in . env file")
        
        self.base_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        self.weather_url = "http://api.openweathermap.org/data/2.5/weather"
        
        print("✅ AQI Data Collector initialized")
    
    def get_air_quality(self, lat:  float, lon: float) -> dict:
        """
        Get CURRENT air quality data for specific coordinates
        
        Args:  
            lat:  Latitude
            lon: Longitude
            
        Returns:
            Dictionary with air quality data
        """
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self. api_key
        }
        
        try:  
            response = requests.get(self.base_url, params=params, timeout=10)
            response. raise_for_status()
            data = response.json()
            
            if 'list' in data and len(data['list']) > 0:
                return data['list'][0]
            else:  
                raise ValueError("No air quality data available")
                
        except requests. exceptions.RequestException as e:
            print(f"❌ Error fetching air quality data: {e}")
            return None
    
    def get_weather(self, lat: float, lon:  float) -> dict:
        """
        Get CURRENT weather data for specific coordinates
        
        Args: 
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dictionary with weather data
        """
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.weather_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching weather data: {e}")
            return None
    
    def collect_data(self, locations: list) -> pd.DataFrame:
        """
        Collect CURRENT air quality and weather data for multiple locations
        
        Args: 
            locations: List of dicts with 'name', 'lat', 'lon'
            
        Returns:
            DataFrame with collected data
        """
        collected_data = []
        
        print(f"\n📡 Collecting LIVE data for {len(locations)} location(s)...")
        
        for location in locations:
            name = location['name']
            lat = location['lat']
            lon = location['lon']
            
            print(f"   📍 {name} ({lat}, {lon})...")
            
            # Get air quality data
            aqi_data = self.get_air_quality(lat, lon)
            if not aqi_data:
                print(f"      ⚠️ Skipping {name} - no AQI data")
                continue
            
            # Get weather data
            weather_data = self.get_weather(lat, lon)
            if not weather_data:
                print(f"      ⚠️ Skipping {name} - no weather data")
                continue
            
            # Extract relevant features
            record = {
                'location': name,
                'latitude': lat,
                'longitude':  lon,
                'timestamp': datetime.now(),
                
                # Air quality components
                'aqi': aqi_data.get('main', {}).get('aqi', None),
                'pm2_5': aqi_data.get('components', {}).get('pm2_5', None),
                'pm10': aqi_data. get('components', {}).get('pm10', None),
                'no2': aqi_data.get('components', {}).get('no2', None),
                'o3': aqi_data.get('components', {}).get('o3', None),
                'so2': aqi_data.get('components', {}).get('so2', None),
                'co': aqi_data.get('components', {}).get('co', None),
                'nh3': aqi_data. get('components', {}).get('nh3', None),
                'no':  aqi_data.get('components', {}).get('no', None),
                
                # Weather data
                'temperature': weather_data.get('main', {}).get('temp', None),
                'feels_like': weather_data.get('main', {}).get('feels_like', None),
                'pressure': weather_data.get('main', {}).get('pressure', None),
                'humidity': weather_data.get('main', {}).get('humidity', None),
                'wind_speed': weather_data.get('wind', {}).get('speed', None),
                'wind_deg': weather_data.get('wind', {}).get('deg', None),
                'clouds': weather_data.get('clouds', {}).get('all', None),
                'weather_main': weather_data.get('weather', [{}])[0].get('main', None),
                'weather_description': weather_data. get('weather', [{}])[0].get('description', None),
            }
            
            collected_data.append(record)
            print(f"      ✅ AQI: {record['aqi']}, PM2.5: {record['pm2_5']}, Temp: {record['temperature']}°C")
        
        df = pd.DataFrame(collected_data)
        print(f"\n✅ Collected {len(df)} record(s)")
        
        return df
    
    def collect_and_save(self, locations: list, feature_store: AQIFeatureStore = None) -> pd.DataFrame:
        """
        Collect CURRENT data and save to feature store
        
        Args:  
            locations: List of locations to collect data for
            feature_store: AQIFeatureStore instance (optional, will create if not provided)
            
        Returns:  
            DataFrame with collected data
        """
        # Collect data
        df = self.collect_data(locations)
        
        if df.empty:
            print("⚠️ No data collected, skipping feature store save")
            return df
        
        # Save to feature store
        close_fs = False
        if feature_store is None:
            feature_store = AQIFeatureStore()
            close_fs = True
        
        try: 
            batch_id = feature_store.save_raw_features(df, source="openweathermap_live")
            print(f"💾 Saved to feature store with batch_id: {batch_id}")
        finally:
            if close_fs: 
                feature_store.close()
        
        return df


# Karachi location
KARACHI = [
    {'name': 'Karachi', 'lat':  24.8607, 'lon': 67.0011},
]


def main():
    """Main function to collect CURRENT data for Karachi"""
    print("="*60)
    print("🌍 LIVE AQI Data Collection - KARACHI")
    print("="*60)
    
    # Initialize collector
    collector = AQIDataCollector()
    
    # Initialize feature store
    with AQIFeatureStore() as fs:
        # Collect and save data
        df = collector. collect_and_save(KARACHI, feature_store=fs)
        
        # Display summary
        if not df.empty:
            print("\n" + "="*60)
            print("📊 Karachi AQI Data Summary")
            print("="*60)
            print(f"Location: {df['location'].iloc[0]}")
            print(f"Timestamp: {df['timestamp'].iloc[0]}")
            print(f"AQI: {df['aqi'].iloc[0]}")
            print(f"PM2.5: {df['pm2_5'].iloc[0]} μg/m³")
            print(f"PM10: {df['pm10']. iloc[0]} μg/m³")
            print(f"Temperature: {df['temperature'].iloc[0]}°C")
            print(f"Humidity: {df['humidity'].iloc[0]}%")
            print(f"Wind Speed: {df['wind_speed'].iloc[0]} m/s")
            
            print("\n📋 Complete Data:")
            print(df. T)  # Transpose for better readability
            
            print("\n💾 Data saved to MongoDB Feature Store!")
        else:
            print("\n⚠️ No data collected")
    
    print("\n✅ Data collection complete!")


if __name__ == "__main__":
    main()