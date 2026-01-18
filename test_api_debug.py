from dotenv import load_dotenv
import os
import requests

print("="*50)
print("🚀 Starting API Test")
print("="*50)

# Load environment variables
load_dotenv()
print("✅ Environment loaded")

API_KEY = os.getenv('OPENWEATHER_API_KEY')
LAT = os.getenv('LAT', '24.8607')
LON = os.getenv('LON', '67.0011')
CITY = os.getenv('CITY', 'Karachi')

print(f"✅ City: {CITY}")
print(f"✅ Latitude: {LAT}")
print(f"✅ Longitude:  {LON}")
print(f"✅ API Key: {'*' * 8}{API_KEY[-4: ] if API_KEY else 'NOT FOUND'}")

if not API_KEY:
    print("\n❌ ERROR: API key not found!")
    exit()

print("\n" + "="*50)
print("🌍 Calling OpenWeather API...")
print("="*50)

url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
print(f"URL: {url[: 80]}...")

try:
    print("\n⏳ Sending request...  (this may take a few seconds)")
    response = requests.get(url, timeout=10)  # 10 second timeout
    
    print(f"✅ Response received!")
    print(f"📊 Status Code: {response. status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n" + "="*50)
        print("✅ SUCCESS! Data retrieved")
        print("="*50)
        
        if 'list' in data and len(data['list']) > 0:
            pollution = data['list'][0]
            aqi = pollution['main']['aqi']
            pm2_5 = pollution['components']. get('pm2_5', 'N/A')
            pm10 = pollution['components'].get('pm10', 'N/A')
            co = pollution['components'].get('co', 'N/A')
            no2 = pollution['components'].get('no2', 'N/A')
            o3 = pollution['components'].get('o3', 'N/A')
            
            print(f"\n🌍 Air Quality in {CITY}:")
            print(f"   AQI Level: {aqi} (1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor)")
            print(f"   PM2.5: {pm2_5} μg/m³")
            print(f"   PM10: {pm10} μg/m³")
            print(f"   CO: {co} μg/m³")
            print(f"   NO2: {no2} μg/m³")
            print(f"   O3: {o3} μg/m³")
            
            print("\n" + "="*50)
            print("✅ Test completed successfully!")
            print("="*50)
        else:
            print("⚠️ No data in response")
    
    elif response.status_code == 401:
        print("\n❌ ERROR: Invalid API key (401 Unauthorized)")
        print("   Your API key might not be activated yet.")
        print("   Wait 5-10 minutes and try again.")
        print("   Or check:  https://home.openweathermap.org/api_keys")
    
    elif response.status_code == 429:
        print("\n❌ ERROR: Too many requests (429)")
        print("   You've hit the rate limit. Wait a minute and try again.")
    
    else:
        print(f"\n❌ ERROR:  Unexpected status code {response.status_code}")
        print(f"Response: {response.text[: 200]}")

except requests.exceptions. Timeout:
    print("\n❌ ERROR: Request timed out")
    print("   The API didn't respond within 10 seconds.")
    print("   Check your internet connection.")

except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Connection failed")
    print("   Could not connect to OpenWeather API.")
    print("   Check your internet connection.")

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("🏁 Script finished")
print("="*50)