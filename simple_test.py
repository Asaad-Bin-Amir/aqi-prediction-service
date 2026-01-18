print("Script started!")

from dotenv import load_dotenv
print("✅ dotenv imported")

import os
print("✅ os imported")

import requests
print("✅ requests imported")

load_dotenv()
print("✅ . env loaded")

API_KEY = os.getenv('OPENWEATHER_API_KEY')
print(f"✅ API Key loaded: {API_KEY is not None}")

print("\n🎉 All steps completed!")