"""
AQI Prediction Dashboard
Streamlit web interface for real-time AQI forecasting
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

def get_text_color(aqi):
    """Get contrasting text color based on AQI background"""
    if aqi <= 2.5:  # Good/Fair (Green/Yellow) - black text
        return "black"
    else:  # Moderate/Poor/Very Poor (Orange/Red/Purple) - white text
        return "white"

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
            latest = fs.raw_features.find_one(sort=[('timestamp', -1)])
            
            if latest:
                return {
                    'aqi': latest.get('aqi'),
                    'pm2_5': latest.get('pm2_5'),
                    'pm10': latest.get('pm10'),
                    'o3': latest.get('o3'),
                    'no2': latest.get('no2'),
                    'so2': latest.get('so2'),
                    'co': latest.get('co'),
                    'temperature': latest.get('temperature'),
                    'humidity': latest.get('humidity'),
                    'wind_speed': latest.get('wind_speed'),
                    'pressure': latest.get('pressure'),
                    'timestamp': latest.get('timestamp'),
                    'location': latest.get('location', 'Karachi'),
                }
    except Exception as e:
        st.error(f"Database error: {str(e)}")
    return None

def load_historical_data(days=7):
    """Load historical data from MongoDB"""
    try:
        with AQIFeatureStore() as fs:
            cutoff = datetime.now() - timedelta(days=days)
            cursor = fs.raw_features.find(
                {'timestamp': {'$gte': cutoff}}
            ).sort('timestamp', 1)
            
            data = list(cursor)
            if data:
                return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
    return pd.DataFrame()

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
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Information")
        
        with st.expander("ℹ️ AQI Scale Reference", expanded=False):
            st.markdown("### 📊 AQI Scale (1-5)")
            st.markdown("""
            - **1** - Good 🟢
            - **2** - Fair 🟡
            - **3** - Moderate 🟠
            - **4** - Poor 🔴
            - **5** - Very Poor 🟣
            
            This project uses the **European CAQI scale** (1-5)
            from OpenWeather API.
            """)
        
        st.markdown("---")
        st.caption("Data updates every hour")
        st.caption("Models: GradientBoosting + XGBoost")
    
    # Load data
    current = load_latest_data()
    
    if not current:
        st.error("❌ No data available. Please check database connection.")
        return
    
    # ========== CURRENT STATUS ==========
    st.header("📊 Current Air Quality")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AQI", f"{current['aqi']}/5")
        st.caption(get_aqi_category(current['aqi']))
    
    with col2:
        st.metric("PM2.5", f"{current['pm2_5']:.1f} µg/m³")
    
    with col3:
        st.metric("PM10", f"{current['pm10']:.1f} µg/m³")
    
    with col4:
        st.metric("Temperature", f"{current['temperature']:.1f}°C")
    
    # AQI Status Banner with dynamic text color
    aqi_color = get_aqi_color(current['aqi'])
    text_color = get_text_color(current['aqi'])
    
    st.markdown(f"""
    <div style='background-color: {aqi_color}; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;'>
        <h2 style='color: {text_color}; margin: 0;'>{get_aqi_category(current['aqi'])}</h2>
        <p style='color: {text_color}; margin: 10px 0 0 0;'>{get_health_message(current['aqi'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"📍 {current['location']} | ⏰ Last updated: {current['timestamp']}")
    
    # ========== FORECASTS ==========
    st.header("🔮 AQI Forecasts")
    
    models = load_models()
    
    if models:
        predictions = make_predictions(current, models)
        
        if predictions:
            cols = st.columns(3)
            for idx, (horizon, pred) in enumerate(predictions.items()):
                with cols[idx]:
                    hours = horizon.replace('h', '')
                    pred_text_color = get_text_color(pred['aqi'])
                    
                    st.markdown(f"""
                    <div style='background-color: {pred['color']}; padding: 15px; border-radius: 8px; text-align: center;'>
                        <h4 style='color: {pred_text_color}; margin: 0;'>{hours}h Forecast</h4>
                        <h2 style='color: {pred_text_color}; margin: 5px 0;'>{pred['aqi']}/5</h2>
                        <p style='color: {pred_text_color}; margin: 0;'>{pred['category']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("⏳ Collecting more data for predictions...")
    else:
        st.warning("⚠️ Models not available.")
    
    # ========== HISTORICAL TRENDS ==========
    st.header("📈 Historical Trends")
    
    df_hist = load_historical_data(days=7)
    
    if not df_hist.empty:
        # AQI Trend with color zones
        fig_aqi = go.Figure()
        
        # Add colored zones
        fig_aqi.add_hrect(y0=0, y1=1.5, fillcolor="green", opacity=0.1, line_width=0)
        fig_aqi.add_hrect(y0=1.5, y1=2.5, fillcolor="yellow", opacity=0.1, line_width=0)
        fig_aqi.add_hrect(y0=2.5, y1=3.5, fillcolor="orange", opacity=0.1, line_width=0)
        fig_aqi.add_hrect(y0=3.5, y1=4.5, fillcolor="red", opacity=0.1, line_width=0)
        fig_aqi.add_hrect(y0=4.5, y1=5.5, fillcolor="purple", opacity=0.1, line_width=0)
        
        # Add AQI line
        fig_aqi.add_trace(go.Scatter(
            x=df_hist['timestamp'],
            y=df_hist['aqi'],
            mode='lines+markers',
            name='AQI',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))
        
        fig_aqi.update_layout(
            title='AQI Trend (Last 7 Days)',
            xaxis_title='Time',
            yaxis_title='AQI (1-5 scale)',
            hovermode='x unified',
            yaxis=dict(range=[0, 6]),
            height=400
        )
        st.plotly_chart(fig_aqi, use_container_width=True)
        
        # Pollutants & Weather
        col1, col2 = st.columns(2)
        
        with col1:
            # Particulate Matter
            fig_pm = go.Figure()
            fig_pm.add_trace(go.Scatter(
                x=df_hist['timestamp'], 
                y=df_hist['pm2_5'], 
                name='PM2.5',
                line=dict(color='#FF6B6B', width=2)
            ))
            fig_pm.add_trace(go.Scatter(
                x=df_hist['timestamp'], 
                y=df_hist['pm10'], 
                name='PM10',
                line=dict(color='#4ECDC4', width=2)
            ))
            fig_pm.update_layout(
                title='Particulate Matter',
                xaxis_title='Time',
                yaxis_title='µg/m³',
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig_pm, use_container_width=True)
        
        with col2:
            # Temperature & Humidity
            fig_weather = go.Figure()
            fig_weather.add_trace(go.Scatter(
                x=df_hist['timestamp'], 
                y=df_hist['temperature'], 
                name='Temperature (°C)',
                line=dict(color='#FF9F40', width=2)
            ))
            fig_weather.add_trace(go.Scatter(
                x=df_hist['timestamp'], 
                y=df_hist['humidity'], 
                name='Humidity (%)',
                yaxis='y2',
                line=dict(color='#4BC0C0', width=2)
            ))
            fig_weather.update_layout(
                title='Weather Conditions',
                xaxis_title='Time',
                yaxis_title='Temperature (°C)',
                yaxis2=dict(
                    title='Humidity (%)', 
                    overlaying='y', 
                    side='right'
                ),
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig_weather, use_container_width=True)
        
        # Additional Pollutants
        st.subheader("🧪 Other Pollutants")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gases (NO2, O3, SO2)
            fig_gases = go.Figure()
            if 'no2' in df_hist.columns:
                fig_gases.add_trace(go.Scatter(
                    x=df_hist['timestamp'], 
                    y=df_hist['no2'], 
                    name='NO₂',
                    line=dict(color='#FF6384', width=2)
                ))
            if 'o3' in df_hist.columns:
                fig_gases.add_trace(go.Scatter(
                    x=df_hist['timestamp'], 
                    y=df_hist['o3'], 
                    name='O₃',
                    line=dict(color='#36A2EB', width=2)
                ))
            if 'so2' in df_hist.columns:
                fig_gases.add_trace(go.Scatter(
                    x=df_hist['timestamp'], 
                    y=df_hist['so2'], 
                    name='SO₂',
                    line=dict(color='#FFCE56', width=2)
                ))
            fig_gases.update_layout(
                title='Gaseous Pollutants',
                xaxis_title='Time',
                yaxis_title='µg/m³',
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig_gases, use_container_width=True)
        
        with col2:
            # Wind Speed & Pressure
            fig_wind = go.Figure()
            fig_wind.add_trace(go.Scatter(
                x=df_hist['timestamp'], 
                y=df_hist['wind_speed'], 
                name='Wind Speed (m/s)',
                line=dict(color='#9966FF', width=2)
            ))
            fig_wind.add_trace(go.Scatter(
                x=df_hist['timestamp'], 
                y=df_hist['pressure'], 
                name='Pressure (hPa)',
                yaxis='y2',
                line=dict(color='#FF9F40', width=2)
            ))
            fig_wind.update_layout(
                title='Wind & Pressure',
                xaxis_title='Time',
                yaxis_title='Wind Speed (m/s)',
                yaxis2=dict(
                    title='Pressure (hPa)', 
                    overlaying='y', 
                    side='right'
                ),
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig_wind, use_container_width=True)
        
        # Statistics
        st.subheader("📊 7-Day Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_aqi = df_hist['aqi'].mean()
            st.metric("Average AQI", f"{avg_aqi:.1f}/5", get_aqi_category(avg_aqi))
        
        with col2:
            max_aqi = df_hist['aqi'].max()
            st.metric("Peak AQI", f"{max_aqi:.1f}/5", "Highest")
        
        with col3:
            avg_pm25 = df_hist['pm2_5'].mean()
            st.metric("Avg PM2.5", f"{avg_pm25:.1f} µg/m³")
        
        with col4:
            avg_temp = df_hist['temperature'].mean()
            st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
    
    else:
        st.info("📊 Collecting historical data... Check back soon!")
    
    # Footer
    st.markdown("---")
    st.caption("Data source: OpenWeather API | AQI Scale: 1-5 (European CAQI)")
    st.caption("Models: GradientBoosting & XGBoost | MongoDB Atlas")

if __name__ == "__main__":
    main()