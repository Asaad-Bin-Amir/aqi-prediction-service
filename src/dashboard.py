"""
AQI Prediction Dashboard
Streamlit web interface for real-time AQI forecasting
AQI Scale: 1-5 (OpenWeather)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os

from src.feature_store import AQIFeatureStore
from src.feature_engineering import engineer_all_features


# Page config
st.set_page_config(
    page_title="AQI Prediction Service",
    page_icon="🌍",
    layout="wide"
)


def get_aqi_color(aqi):
    """Get color for AQI value (1-5 scale)"""
    if aqi <= 1.5:
        return "#00E400"  # Green (Good)
    elif aqi <= 2.5:
        return "#FFFF00"  # Yellow (Fair)
    elif aqi <= 3.5:
        return "#FF7E00"  # Orange (Moderate)
    elif aqi <= 4.5:
        return "#FF0000"  # Red (Poor)
    else:
        return "#8F3F97"  # Purple (Very Poor)


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


def load_latest_data():
    """Load latest AQI data from MongoDB"""
    with AQIFeatureStore() as fs:
        latest = fs.raw_features.find_one(sort=[('timestamp', -1)])
        
        if latest:
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
    return None


def load_historical_data(days=7):
    """Load historical AQI data"""
    with AQIFeatureStore() as fs:
        cutoff = datetime.now() - timedelta(days=days)
        data = list(fs.raw_features.find(
            {'timestamp': {'$gte': cutoff}}
        ).sort('timestamp', 1))
        
        if data:
            return pd.DataFrame(data)
    return pd.DataFrame()


def load_models():
    """Load trained forecast models"""
    models = {}
    
    for horizon in ['24h', '48h', '72h']:
        model_path = f'models/aqi_model_{horizon}.joblib'
        metadata_path = f'models/aqi_model_{horizon}_metadata.joblib'
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            models[horizon] = {
                'model': joblib.load(model_path),
                'metadata': joblib.load(metadata_path)
            }
    
    return models


def make_predictions(current_data, models):
    """Generate forecasts using trained models"""
    predictions = {}
    
    if not models:
        return predictions
    
    # Prepare feature data
    with AQIFeatureStore() as fs:
        data = list(fs.raw_features.find({}).sort('timestamp', -1).limit(100))
    
    if len(data) < 50:
        st.warning("⚠️ Insufficient data for predictions (need at least 50 records)")
        return predictions
    
    df = pd.DataFrame(data).sort_values('timestamp')
    
    for horizon, model_info in models.items():
        try:
            # Engineer features
            df_eng = engineer_all_features(df.copy(), horizons=[horizon])
            
            # Get latest features
            feature_cols = model_info['metadata']['feature_cols']
            X = df_eng[feature_cols].iloc[-1:].fillna(0)
            
            # Predict
            pred = model_info['model'].predict(X)[0]
            pred = max(1, min(5, pred))  # Clip to 1-5
            
            predictions[horizon] = {
                'aqi': round(pred, 1),
                'category': get_aqi_category(pred),
                'color': get_aqi_color(pred)
            }
        except Exception as e:
            st.error(f"Prediction error for {horizon}: {e}")
    
    return predictions


# Main Dashboard
def main():
    st.title("🌍 AQI Prediction Service")
    st.caption("Real-time Air Quality Index forecasting powered by Machine Learning")
    
    # Load data
    current = load_latest_data()
    models = load_models()
    
    if not current:
        st.error("❌ No data available. Please run data collection first.")
        return
    
    # Current Status
    st.header("📊 Current Air Quality")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="AQI",
            value=f"{current['aqi']}/5",
            delta=get_aqi_category(current['aqi'])
        )
    
    with col2:
        st.metric(
            label="PM2.5",
            value=f"{current['pm2_5']:.1f} µg/m³"
        )
    
    with col3:
        st.metric(
            label="PM10",
            value=f"{current['pm10']:.1f} µg/m³"
        )
    
    with col4:
        st.metric(
            label="Temperature",
            value=f"{current['temperature']:.1f}°C"
        )
    
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
            st.info("⏳ Collecting more data for accurate predictions...")
    else:
        st.warning("⚠️ No trained models available. Please run training pipeline.")
    
    # Historical Trends
    st.header("📈 Historical Trends")
    
    df_hist = load_historical_data(days=7)
    
    if not df_hist.empty:
        # AQI trend
        fig_aqi = go.Figure()
        fig_aqi.add_trace(go.Scatter(
            x=df_hist['timestamp'],
            y=df_hist['aqi'],
            mode='lines+markers',
            name='AQI',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=6)
        ))
        fig_aqi.update_layout(
            title='AQI Trend (Last 7 Days)',
            xaxis_title='Time',
            yaxis_title='AQI (1-5 scale)',
            hovermode='x unified',
            yaxis=dict(range=[0, 6])
        )
        st.plotly_chart(fig_aqi, use_container_width=True)
        
        # Pollutants
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pm = go.Figure()
            fig_pm.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['pm2_5'], name='PM2.5'))
            fig_pm.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['pm10'], name='PM10'))
            fig_pm.update_layout(title='Particulate Matter', xaxis_title='Time', yaxis_title='µg/m³')
            st.plotly_chart(fig_pm, use_container_width=True)
        
        with col2:
            fig_weather = go.Figure()
            fig_weather.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['temperature'], name='Temperature'))
            fig_weather.add_trace(go.Scatter(x=df_hist['timestamp'], y=df_hist['humidity'], name='Humidity', yaxis='y2'))
            fig_weather.update_layout(
                title='Weather Conditions',
                xaxis_title='Time',
                yaxis_title='Temperature (°C)',
                yaxis2=dict(title='Humidity (%)', overlaying='y', side='right')
            )
            st.plotly_chart(fig_weather, use_container_width=True)
    else:
        st.info("📊 Collecting historical data... Check back soon!")
    
    # Footer
    st.markdown("---")
    st.caption("Data source: OpenWeather API | AQI Scale: 1-5 (European Index)")


if __name__ == "__main__":
    main()