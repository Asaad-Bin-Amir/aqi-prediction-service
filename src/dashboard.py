"""
AQI Prediction Dashboard
Streamlit web interface for real-time AQI forecasting
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import os

from feature_store import AQIFeatureStore
from model_registry import ModelRegistry

# Page config
st.set_page_config(
    page_title="AQI Prediction Service",
    page_icon="🌍",
    layout="wide"
)

def get_aqi_color(aqi):
    """Get color for AQI value (1-5 scale)"""
    if aqi <= 1.5:
        return "#00E400"
    elif aqi <= 2.5:
        return "#FFFF00"
    elif aqi <= 3.5:
        return "#FF7E00"
    elif aqi <= 4.5:
        return "#FF0000"
    else:
        return "#8F3F97"

def get_aqi_category(aqi):
    """Get AQI category name (1-5 scale)"""
    if aqi <= 1.5:
        return "Good"
    elif aqi <= 2.5:
        return "Fair"
    elif aqi <= 3.5:
        return "Moderate"
    elif aqi <= 4.5:
        return "Poor"
    else:
        return "Very Poor"

def get_health_message(aqi):
    """Get health recommendation for AQI level"""
    if aqi <= 1.5:
        return "Air quality is satisfactory, air pollution poses little or no risk."
    elif aqi <= 2.5:
        return "Air quality is acceptable. However, there may be a risk for some people."
    elif aqi <= 3.5:
        return "Members of sensitive groups may experience health effects."
    elif aqi <= 4.5:
        return "Everyone may begin to experience health effects."
    else:
        return "Health warning of emergency conditions. Everyone is more likely to be affected."

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features to match training pipeline"""
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 10) | 
                          (df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'pm2_5_lag_{lag}h'] = df['pm2_5'].shift(lag)
        df[f'pm10_lag_{lag}h'] = df['pm10'].shift(lag)
        df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)
    
    # Rolling features
    for window in [3, 6, 12, 24, 48]:
        df[f'pm2_5_rolling_mean_{window}h'] = df['pm2_5'].rolling(window, min_periods=1).mean()
        df[f'pm2_5_rolling_std_{window}h'] = df['pm2_5'].rolling(window, min_periods=1).std()
        df[f'pm2_5_rolling_max_{window}h'] = df['pm2_5'].rolling(window, min_periods=1).max()
        df[f'pm2_5_rolling_min_{window}h'] = df['pm2_5'].rolling(window, min_periods=1).min()
        df[f'aqi_rolling_mean_{window}h'] = df['aqi'].rolling(window, min_periods=1).mean()
        df[f'aqi_rolling_std_{window}h'] = df['aqi'].rolling(window, min_periods=1).std()
    
    # Rate of change
    df['pm2_5_change_1h'] = df['pm2_5'].diff(1)
    df['pm2_5_change_3h'] = df['pm2_5'].diff(3)
    df['pm2_5_change_6h'] = df['pm2_5'].diff(6)
    
    # Interactions
    df['pm2_5_x_wind'] = df['pm2_5'] * df['wind_speed']
    df['pm2_5_x_humidity'] = df['pm2_5'] * df['humidity']
    df['pm2_5_x_temp'] = df['pm2_5'] * df['temperature']
    df['wind_x_humidity'] = df['wind_speed'] * df['humidity']
    df['temp_x_humidity'] = df['temperature'] * df['humidity']
    df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1)
    
    # Weather changes
    df['temp_change'] = df['temperature'].diff(1)
    df['wind_change'] = df['wind_speed'].diff(1)
    df['pressure_change'] = df['pressure'].diff(1)
    
    return df

@st.cache_resource(ttl=3600)
def load_models():
    """Load models from MongoDB"""
    models = {}
    try:
        with ModelRegistry() as registry:
            for horizon in ['24h', '48h', '72h']:
                model, metadata = registry.get_production_model(f'aqi_forecast_{horizon}')
                if model and metadata:
                    models[horizon] = {'model': model, 'metadata': metadata}
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
    return models

def load_latest_data():
    """Load latest AQI data from MongoDB"""
    try:
        with AQIFeatureStore() as fs:
            # Force fresh query - no caching
            latest = fs.raw_features.find_one(
                sort=[('timestamp', -1)]
            )
            
            if latest:
                print(f"DEBUG: Loaded AQI = {latest.get('aqi')} from {latest.get('timestamp')}")  # Debug
                return {
                    'aqi': latest.get('aqi'),
                    'pm2_5': latest.get('pm2_5'),
                    'pm10': latest.get('pm10'),
                    'o3': latest.get('o3'),
                    'temperature': latest.get('temperature'),
                    'humidity': latest.get('humidity'),
                    'wind_speed': latest.get('wind_speed'),
                    'timestamp': latest.get('timestamp'),
                    'location': latest.get('location', 'Karachi'),
                }
    except Exception as e:
        st.error(f"Database error: {str(e)}")
    return None

def make_predictions(current_data, models):
    """Generate predictions"""
    predictions = {}
    if not models:
        return predictions
    
    try:
        with AQIFeatureStore() as fs:
            data = list(fs.raw_features.find({}).sort('timestamp', -1).limit(100))
        
        if len(data) < 50:
            return predictions
        
        df = pd.DataFrame(data).sort_values('timestamp').reset_index(drop=True)
        df_eng = engineer_features(df.copy())
        
        for horizon, model_info in models.items():
            try:
                feature_cols = model_info['metadata']['feature_cols']
                X = df_eng[feature_cols].iloc[-1:].fillna(0)
                pred = model_info['model'].predict(X)[0]
                pred = max(1, min(5, pred))
                
                predictions[horizon] = {
                    'aqi': round(pred, 1),
                    'category': get_aqi_category(pred),
                    'color': get_aqi_color(pred)
                }
            except Exception as e:
                continue
    except Exception as e:
        pass
    
    return predictions

def main():
    st.title("🌍 AQI Prediction Service")
    st.caption("Real-time Air Quality Index forecasting powered by Machine Learning")
    
    # Load data
    current = load_latest_data()
    
    if not current:
        st.error("❌ No data available. Please check database connection.")
        return
    
    # Current Status
    st.header("📊 Current Air Quality")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AQI", f"{current['aqi']}/5", get_aqi_category(current['aqi']))
    
    with col2:
        st.metric("PM2.5", f"{current['pm2_5']:.1f} µg/m³")
    
    with col3:
        st.metric("PM10", f"{current['pm10']:.1f} µg/m³")
    
    with col4:
        st.metric("Temperature", f"{current['temperature']:.1f}°C")
    
    # AQI Indicator
    aqi_color = get_aqi_color(current['aqi'])
    st.markdown(f"""
    <div style='background-color: {aqi_color}; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2 style='color: white; margin: 0;'>{get_aqi_category(current['aqi'])}</h2>
        <p style='color: white; margin: 10px 0 0 0;'>{get_health_message(current['aqi'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"📍 {current['location']} | ⏰ Last updated: {current['timestamp']}")
    
    # Forecasts
    st.header("🔮 AQI Forecasts")
    
    models = load_models()
    
    if models:
        predictions = make_predictions(current, models)
        
        if predictions:
            cols = st.columns(3)
            for idx, (horizon, pred) in enumerate(predictions.items()):
                with cols[idx]:
                    hours = horizon.replace('h', '')
                    st.markdown(f"""
                    <div style='background-color: {pred['color']}; padding: 15px; border-radius: 8px; text-align: center;'>
                        <h4 style='color: white; margin: 0;'>{hours}h Forecast</h4>
                        <h2 style='color: white; margin: 5px 0;'>{pred['aqi']}/5</h2>
                        <p style='color: white; margin: 0;'>{pred['category']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("⏳ Collecting more data for predictions...")
    else:
        st.warning("⚠️ Models not available. Training in progress...")
    
    # Footer
    st.markdown("---")
    st.caption("Data source: OpenWeather API | AQI Scale: 1-5 (European Index)")

if __name__ == "__main__":
    main()