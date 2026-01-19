from src.utils import *

# Test AQI calculation
pm25 = 21.32
aqi = calculate_aqi_from_pm25(pm25)
category = get_aqi_category(aqi)

print(f"PM2.5: {pm25} μg/m³")
print(f"AQI: {aqi}")
print(f"Category: {category['category']}")
print(f"Color: {category['color']}")
print(f"Message: {category['message']}")