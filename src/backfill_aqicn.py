import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

AQICN_API_TOKEN = os.getenv('AQICN_API_TOKEN')

def find_karachi_stations():
    """Find all air quality monitoring stations in Karachi"""
    print("🔍 Searching for Karachi monitoring stations...")
    
    url = f"https://api.waqi.info/search/? token={AQICN_API_TOKEN}&keyword=karachi"
    
    try: 
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'ok':
            print(f"❌ API Error: {data.get('data', 'Unknown error')}")
            return []
        
        stations = data['data']
        print(f"✅ Found {len(stations)} stations:")
        
        for i, station in enumerate(stations, 1):
            print(f"   {i}. {station['station']['name']} - AQI: {station. get('aqi', 'N/A')}")
        
        return stations
    
    except Exception as e:
        print(f"❌ Error:  {e}")
        return []

def get_station_current_data(station_id):
    """Get current data from a specific station"""
    print(f"\n📡 Fetching data from station ID: {station_id}")
    
    url = f"https://api.waqi.info/feed/@{station_id}/? token={AQICN_API_TOKEN}"
    
    try:
        response = requests. get(url, timeout=10)
        response.raise_for_status()
        data = response. json()
        
        if data['status'] != 'ok': 
            print(f"❌ Error: {data.get('data')}")
            return None
        
        return data['data']
    
    except Exception as e: 
        print(f"❌ Error: {e}")
        return None

def extract_pollutant_data(station_data):
    """Extract pollutant readings from station data"""
    
    iaqi = station_data.get('iaqi', {})
    
    record = {
        'timestamp': pd.to_datetime(station_data['time']['s']),
        'city': station_data. get('city', {}).get('name', 'Karachi'),
        'aqi': station_data.get('aqi'),
    }
    
    # Add coordinates if available
    geo = station_data.get('city', {}).get('geo', [])
    if len(geo) == 2:
        record['lat'] = geo[0]
        record['lon'] = geo[1]
    
    # Extract pollutants
    pollutants = {
        'pm25': 'pm2_5',
        'pm10':  'pm10',
        'no2': 'no2',
        'so2': 'so2',
        'o3': 'o3',
        'co': 'co',
        't': 'temp',
        'h': 'humidity',
        'p': 'pressure',
        'w': 'wind_speed'
    }
    
    for api_key, our_key in pollutants.items():
        if api_key in iaqi:
            record[our_key] = iaqi[api_key]. get('v')
    
    return record

def calculate_aqi_from_pm25(pm25):
    """
    Calculate actual AQI value (0-500) from PM2.5 using EPA formula
    Not just category (1-5), but the actual numeric AQI
    """
    # EPA AQI breakpoints for PM2.5 (24-hour average)
    # Format: (PM2.5_low, PM2.5_high, AQI_low, AQI_high)
    breakpoints = [
        (0.0, 12.0, 0, 50),           # Good
        (12.1, 35.4, 51, 100),         # Moderate
        (35.5, 55.4, 101, 150),        # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200),       # Unhealthy
        (150.5, 250.4, 201, 300),      # Very Unhealthy
        (250.5, 350.4, 301, 400),      # Hazardous
        (350.5, 500.4, 401, 500),      # Hazardous+
    ]
    
    for pm_low, pm_high, aqi_low, aqi_high in breakpoints:
        if pm_low <= pm25 <= pm_high:
            # Linear interpolation formula
            aqi = ((aqi_high - aqi_low) / (pm_high - pm_low)) * (pm25 - pm_low) + aqi_low
            return round(aqi, 1)  # Return decimal AQI value
    
    # If PM2.5 is above 500.4, cap at AQI 500
    if pm25 > 500.4:
        return 500.0
    
    # If PM2.5 is 0 or negative
    return 0.0

def generate_synthetic_historical(base_record, days=90):
    """Generate realistic synthetic historical data based on Karachi patterns"""
    print(f"\n🔧 Generating {days} days of synthetic historical data...")
    print("   Based on Karachi's seasonal and daily AQI patterns")
    
    records = []
    end_time = datetime.now()
    
    # Get base values - use PM2.5 to calculate AQI
    base_pm25 = base_record.get('pm2_5', 125)  # Real Karachi baseline
    base_pm10 = base_record.get('pm10', 185)
    base_temp = base_record.get('temp', 26)
    base_humidity = base_record.get('humidity', 58)
    
    for hour in range(days * 24):
        timestamp = end_time - timedelta(hours=hour)
        
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        month = timestamp.month
        
        # Karachi-specific patterns
        # Winter (Nov-Feb): High pollution (150-250 PM2.5)
        # Monsoon (Jun-Aug): Lower pollution (50-100 PM2.5)
        # Spring/Fall: Moderate (100-180 PM2.5)
        
        # Seasonal factor
        if month in [11, 12, 1, 2]:  # Winter - very high pollution
            seasonal = 1.5
        elif month in [6, 7, 8]:   # Monsoon - rain clears air
            seasonal = 0.5
        elif month in [3, 4, 5]:  # Spring - dusty, hot
            seasonal = 1.3
        else:  # Autumn
            seasonal = 1.1
        
        # Daily pattern (rush hours 7-10 AM, 5-9 PM)
        if hour_of_day in [7, 8, 9, 17, 18, 19, 20]:
            hourly = 1.4
        elif hour_of_day in [2, 3, 4, 5]:   # Early morning - clearest
            hourly = 0.6
        else:
            hourly = 1.0
        
        # Weekday vs weekend
        weekday = 1.2 if day_of_week < 5 else 0.8
        
        # Random variation
        noise = np.random.normal(1.0, 0.15)
        
        combined_factor = seasonal * hourly * weekday * noise
        
        # Calculate PM2.5 with realistic variation
        pm25_value = max(5, base_pm25 * combined_factor * np.random.uniform(0.6, 1.4))
        
        # ✅ Calculate REAL AQI from PM2.5 (not just category 1-5)
        aqi_value = calculate_aqi_from_pm25(pm25_value)
        
        record = {
            'timestamp': timestamp,
            'city': 'Karachi',
            'lat': base_record.get('lat', 24.8607),
            'lon': base_record.get('lon', 67.0011),
            'aqi': aqi_value,  # ✅ Now this will be a proper AQI value (0-500)
            'pm2_5': pm25_value,
            'pm10': max(10, pm25_value * 1.3 * np.random.uniform(0.9, 1.2)),
            'co': np.random.uniform(400, 1200) * combined_factor,
            'no2': np.random.uniform(20, 90) * combined_factor,
            'o3': np.random.uniform(30, 100) * (2 - combined_factor),
            'so2': np.random.uniform(10, 45) * combined_factor,
            'temp': base_temp + np.random.normal(0, 3) + (5 if month in [5, 6] else -8 if month in [12, 1] else 0),
            'humidity': max(20, min(90, base_humidity + np.random.normal(0, 15))),
            'pressure': 1013 + np.random.normal(0, 5),
            'wind_speed':  max(0, np.random.exponential(3)),
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Generated {len(df)} hourly records")
    print(f"📊 Date range: {df['timestamp'].min()} to {df['timestamp']. max()}")
    
    return df

def save_to_csv(df):
    """Save data to CSV file"""
    os.makedirs('data', exist_ok=True)
    
    filename = f'data/karachi_aqi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(filename, index=False)
    
    print(f"\n💾 Saved to: {filename}")
    print(f"📊 Total records: {len(df)}")
    print(f"📊 Columns: {list(df.columns)}")
    
    # Show sample statistics
    print(f"\n📈 Data Statistics:")
    print(f"   AQI range: {df['aqi'].min():.1f} - {df['aqi'].max():.1f}")
    print(f"   PM2.5 range: {df['pm2_5'].min():.1f} - {df['pm2_5'].max():.1f} μg/m³")
    print(f"   PM10 range: {df['pm10']. min():.1f} - {df['pm10'].max():.1f} μg/m³")
    print(f"   Humidity range: {df['humidity'].min():.1f} - {df['humidity'].max():.1f}%")
    
    # Show AQI distribution
    print(f"\n📊 AQI Distribution:")
    aqi_bins = [0, 50, 100, 150, 200, 300, 500]
    aqi_labels = ['Good (0-50)', 'Moderate (51-100)', 'USG (101-150)', 'Unhealthy (151-200)', 'Very Unhealthy (201-300)', 'Hazardous (301+)']
    df['aqi_category'] = pd.cut(df['aqi'], bins=aqi_bins, labels=aqi_labels)
    print(df['aqi_category'].value_counts().sort_index())
    
    return filename

if __name__ == "__main__": 
    print("="*60)
    print("🌍 AQICN Data Backfill for Karachi (3 months)")
    print("="*60)
    
    if not AQICN_API_TOKEN:
        print("\n❌ AQICN_API_TOKEN not found in . env file")
        print("\n📝 Get your free token:")
        print("   1. Go to:  https://aqicn.org/data-platform/token/")
        print("   2. Fill the form and submit")
        print("   3. Copy token and add to .env file:")
        print("      AQICN_API_TOKEN=your_token_here")
        exit(1)
    
    # Step 1: Find Karachi stations
    stations = find_karachi_stations()
    
    if not stations:
        print("\n⚠️ No stations found, using default Karachi data")
        base_record = {
            'aqi': 177,
            'pm2_5': 125,
            'pm10':  185,
            'temp':  26,
            'humidity':  58,
            'lat':  24.8607,
            'lon': 67.0011
        }
    else: 
        # Step 2: Get data from first (main) station
        station_id = stations[0]['uid']
        station_data = get_station_current_data(station_id)
        
        if station_data:
            base_record = extract_pollutant_data(station_data)
            
            print(f"\n✅ Current readings from {station_data['city']['name']}:")
            print(f"   AQI: {base_record. get('aqi', 'N/A')}")
            print(f"   PM2.5: {base_record.get('pm2_5', 'N/A')} μg/m³")
            print(f"   PM10: {base_record. get('pm10', 'N/A')} μg/m³")
            print(f"   Temperature: {base_record. get('temp', 'N/A')}°C")
            print(f"   Humidity: {base_record.get('humidity', 'N/A')}%")
        else:
            base_record = {
                'aqi': 177,
                'pm2_5': 125,
                'pm10': 185,
                'temp': 26,
                'humidity': 58,
                'lat': 24.8607,
                'lon': 67.0011
            }
    
    # Step 3: Generate 3 months of historical data
    df = generate_synthetic_historical(base_record, days=90)
    
    # Step 4: Save to CSV
    filename = save_to_csv(df)
    
    print("\n" + "="*60)
    print("✅ Backfill complete!")
    print("="*60)
    print(f"\n📝 Next step:  Train your models")
    print(f"   python src/training_pipeline.py")