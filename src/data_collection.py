"""
AQICN Data Collection with PM10 Estimation
Uses AQICN for accurate AQI/PM2.5, estimates PM10 based on typical ratios
"""
import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from feature_store import AQIFeatureStore

load_dotenv()

# AQICN API Configuration
AQICN_TOKEN = os.getenv('AQICN_API_TOKEN')

# Karachi coordinates
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011


class AQICNCollector:
    """Collect AQI data from AQICN API"""
    
    def __init__(self):
        """Initialize AQICN collector"""
        if not AQICN_TOKEN:
            raise ValueError("❌ AQICN_API_TOKEN not found in .env file!")
        
        self.token = AQICN_TOKEN
        self.base_url = "https://api.waqi.info"
        
        print("="*60)
        print("🌆 AQICN Data Collection - KARACHI")
        print("✅ Using AQICN API (Real monitoring stations)")
        print("="*60)
    
    def get_city_data(self, city_name="karachi"):
        """
        Get AQI data for a city from AQICN
        
        Args:
            city_name: City name (default: karachi)
        
        Returns:
            dict: AQI data including pollutants and weather
        """
        url = f"{self.base_url}/feed/{city_name}/"
        params = {'token': self.token}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'ok':
                raise ValueError(f"API error: {data.get('data', 'Unknown error')}")
            
            return self._parse_aqicn_data(data['data'])
        
        except Exception as e:
            print(f"❌ Error fetching data for {city_name}: {e}")
            return None
    
    def _parse_aqicn_data(self, data):
        """Parse AQICN API response into our format"""
        
        # Extract coordinates
        lat = data['city']['geo'][0]
        lon = data['city']['geo'][1]
        
        # Extract AQI (already in EPA scale!)
        aqi = data['aqi']
        
        # Extract pollutants (iaqi = individual air quality index)
        iaqi = data.get('iaqi', {})
        
        # Get PM2.5 (most important!)
        pm25_aqi = iaqi.get('pm25', {}).get('v', None)
        pm25 = self._aqi_to_pm25(pm25_aqi) if pm25_aqi else None
        
        # Try to get PM10 from AQICN
        pm10_aqi = iaqi.get('pm10', {}).get('v', None)
        pm10_from_api = self._aqi_to_pm10(pm10_aqi) if pm10_aqi else None
        
        # If PM10 not available, estimate it (PM10 typically 2.2x PM2.5 in urban areas)
        if pm10_from_api:
            pm10 = pm10_from_api
            pm10_source = "AQICN"
        elif pm25:
            pm10 = pm25 * 2.2  # Typical ratio for South Asian cities
            pm10_source = "estimated"
        else:
            pm10 = 0
            pm10_source = "default"
        
        # Other pollutants (use AQI values as proxies for concentrations)
        no2 = iaqi.get('no2', {}).get('v', 0)
        o3 = iaqi.get('o3', {}).get('v', 0)
        so2 = iaqi.get('so2', {}).get('v', 0)
        co = iaqi.get('co', {}).get('v', 0)
        
        # Extract weather data
        temp = iaqi.get('t', {}).get('v', None)
        pressure = iaqi.get('p', {}).get('v', None)
        humidity = iaqi.get('h', {}).get('v', None)
        wind = iaqi.get('w', {}).get('v', None)
        
        # Validate humidity (AQICN sometimes gives bad data)
        if humidity and (humidity < 10 or humidity > 100):
            print(f"   ⚠️ Invalid humidity {humidity}%, using default 50%")
            humidity = 50
        
        # Get timestamp
        timestamp = datetime.fromisoformat(data['time']['iso'].replace('Z', '+00:00'))
        
        record = {
            'location': 'Karachi',
            'latitude': lat,
            'longitude': lon,
            'timestamp': timestamp,
            'aqi': aqi,  # EPA AQI scale (0-500)
            'pm2_5': pm25 if pm25 else 0,
            'pm10': pm10,
            'no2': no2,
            'o3': o3,
            'so2': so2,
            'co': co,
            'nh3': 0,  # AQICN doesn't provide
            'no': 0,   # AQICN doesn't provide
            'temperature': temp if temp else 25,  # Default to 25°C
            'feels_like': temp if temp else 25,
            'pressure': pressure if pressure else 1013,
            'humidity': humidity if humidity else 50,
            'wind_speed': wind if wind else 0,
            'wind_deg': 0,
            'clouds': 0,
            'weather_main': 'Unknown',
            'weather_description': f'Data from AQICN station: {data["city"].get("name", "Karachi")}'
        }
        
        print(f"\n📍 Karachi (AQICN Monitoring Station)")
        print(f"   Station: {data['city'].get('name', 'Karachi')}")
        print(f"   Timestamp: {timestamp}")
        print(f"   ✅ AQI: {aqi} (EPA scale 0-500)")
        print(f"   ✅ PM2.5: {pm25:.2f} μg/m³" if pm25 else "   ⚠️ PM2.5: N/A")
        print(f"   {'✅' if pm10_source == 'AQICN' else '📊'} PM10: {pm10:.2f} μg/m³ ({pm10_source})")
        print(f"   🌡️ Temperature: {temp}°C" if temp else "   🌡️ Temperature: N/A (using default)")
        print(f"   💧 Humidity: {humidity}%" if humidity else "   💧 Humidity: N/A (using default)")
        
        return record
    
    def _aqi_to_pm25(self, aqi):
        """Convert PM2.5 AQI to concentration (μg/m³) - EPA breakpoints"""
        if aqi is None:
            return None
        
        breakpoints = [
            (0, 50, 0.0, 12.0),
            (51, 100, 12.1, 35.4),
            (101, 150, 35.5, 55.4),
            (151, 200, 55.5, 150.4),
            (201, 300, 150.5, 250.4),
            (301, 500, 250.5, 500.4)
        ]
        
        for aqi_low, aqi_high, conc_low, conc_high in breakpoints:
            if aqi_low <= aqi <= aqi_high:
                conc = ((aqi - aqi_low) / (aqi_high - aqi_low)) * (conc_high - conc_low) + conc_low
                return conc
        
        return aqi
    
    def _aqi_to_pm10(self, aqi):
        """Convert PM10 AQI to concentration (μg/m³) - EPA breakpoints"""
        if aqi is None:
            return None
        
        breakpoints = [
            (0, 50, 0, 54),
            (51, 100, 55, 154),
            (101, 150, 155, 254),
            (151, 200, 255, 354),
            (201, 300, 355, 424),
            (301, 500, 425, 604)
        ]
        
        for aqi_low, aqi_high, conc_low, conc_high in breakpoints:
            if aqi_low <= aqi <= aqi_high:
                conc = ((aqi - aqi_low) / (aqi_high - aqi_low)) * (conc_high - conc_low) + conc_low
                return conc
        
        return aqi
    
    def collect_and_save(self):
        """Collect data and save to MongoDB"""
        
        # Collect data
        data = self.get_city_data("karachi")
        
        if not data:
            print("❌ Failed to collect data")
            return False
        
        # Save to MongoDB
        with AQIFeatureStore() as fs:
            batch_id = f"aqicn_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add metadata
            data['batch_id'] = batch_id
            data['source'] = 'aqicn'
            data['collection_type'] = 'live'
            
            # Save to raw_features collection
            fs.raw_features.insert_one(data)
            
            print(f"\n💾 Saved to MongoDB")
            print(f"   Batch: {batch_id}")
            print(f"   Collection: raw_features")
            print(f"   AQI: {data['aqi']}")
        
        return True


def main():
    """Main execution"""
    try:
        collector = AQICNCollector()
        
        print("\n🔄 Collecting live data from AQICN...")
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