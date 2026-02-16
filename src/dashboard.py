"""
AQI Prediction Dashboard
Streamlit web interface for real-time AQI forecasting
Loads models from MongoDB GridFS (works on Streamlit Cloud!)
AQI Scale: 1-5 (OpenWeather)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import joblib
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer ALL features to match training pipeline
    Creates 82 features used by optimized models
    """
    # Time features
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
    
    # Rate of change features
    df['pm2_5_change_1h'] = df['pm2_5'].diff(1)
    df['pm2_5_change_3h'] = df['pm2_5'].diff(3)
    df['pm2_5_change_6h'] = df['pm2_5'].diff(6)
    
    # Interaction features
    df['pm2_5_x_wind'] = df['pm2_5'] * df['wind_speed']
    df['pm2_5_x_humidity'] = df['pm2_5'] * df['humidity']
    df['pm2_5_x_temp'] = df['pm2_5'] * df['temperature']
    df['wind_x_humidity'] = df['wind_speed'] * df['humidity']
    df['temp_x_humidity'] = df['temperature'] * df['humidity']
    df['pm2_5_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 1)
    
    # Weather change features
    df['temp_change'] = df['temperature'].diff(1)
    df['wind_change'] = df['wind_speed'].diff(1)
    df['pressure_change'] = df['pressure'].diff(1)
    
    return df


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


@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_models():
    """Load trained forecast models from MongoDB GridFS"""
    models = {}
    
    try:
        with ModelRegistry() as registry:
            for horizon in ['24h', '48h', '72h']:
                model_name = f'aqi_forecast_{horizon}'
                
                # Try production first
                model, metadata = registry.get_production_model(model_name)
                
                # Fallback to latest staging
                if not model:
                    model, metadata = registry.load_model(model_name)
                
                if model and metadata:
                    models[horizon] = {
                        'model': model,
                        'metadata': metadata
                    }
                    stage = metadata.get('stage', 'unknown')
                    print(f"✅ Loaded {horizon} from MongoDB ({stage})")
    
    except Exception as e:
        st.error(f"⚠️ Failed to load from MongoDB: {str(e)}")
        print(f"Error loading models: {e}")
        
        # Fallback to local files (for local development)
        try:
            for horizon in ['24h', '48h', '72h']:
                model_path = f'models/aqi_model_{horizon}.joblib'
                metadata_path = f'models/aqi_model_{horizon}_metadata.joblib'
                
                if os.path.exists(model_path) and os.path.exists(metadata_path):
                    models[horizon] = {
                        'model': joblib.load(model_path),
                        'metadata': joblib.load(metadata_path)
                    }
                    print(f"✅ Loaded {horizon} from local file")
        except Exception as local_err:
            print(f"Local fallback also failed: {local_err}")
    
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
    
    df = pd.DataFrame(data).sort_values('timestamp').reset_index(drop=True)
    
    for horizon, model_info in models.items():
        try:
            # Engineer features (MATCH TRAINING PIPELINE)
            df_eng = engineer_features(df.copy())
            
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
            st.error(f"Prediction error for {horizon}: {str(e)}")
    
    return predictions


# Main Dashboard
def main():
    st.title("🌍 AQI Prediction Service")
    st.caption("Real-time Air Quality Index forecasting powered by Machine Learning")
    
    # ========== SIDEBAR - AQI SCALE REFERENCE ==========
    with st.sidebar:
        st.title("📊 Information")
        
        with st.expander("ℹ️ AQI Scale Reference", expanded=False):
            st.markdown("### 📊 AQI Scale Comparison")
            st.markdown("*Understanding air quality indices*")
            
            # Two-column layout for scale comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**OpenWeather CAQI**")
                st.markdown("*(1-5 scale)*")
                st.markdown("*Used in this project*")
                
                # Visual gradient
                st.markdown("""
                <div style='background: linear-gradient(to bottom, #00e400, #ffff00, #ff7e00, #ff0000, #8f3f97); 
                            padding: 15px; border-radius: 8px; text-align: center; color: white; font-weight: bold;
                            margin-bottom: 10px;'>
                    1<br>↓<br>2<br>↓<br>3<br>↓<br>4<br>↓<br>5
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                - **1** - Good ✅
                - **2** - Fair ⚪
                - **3** - Moderate 🟡
                - **4** - Poor 🟠
                - **5** - Very Poor 🔴
                """)
            
            with col2:
                st.markdown("**US EPA AQI**")
                st.markdown("*(0-500 scale)*")
                st.markdown("*Traditional standard*")
                
                # Visual gradient
                st.markdown("""
                <div style='background: linear-gradient(to bottom, #00e400, #ffff00, #ff7e00, #ff0000, #8f3f97, #7e0023); 
                            padding: 15px; border-radius: 8px; text-align: center; color: white; font-weight: bold;
                            margin-bottom: 10px; font-size: 12px;'>
                    0-50<br>↓<br>51-100<br>↓<br>101-150<br>↓<br>151-200<br>↓<br>201-300<br>↓<br>301-500
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                - **0-50** - Good ✅
                - **51-100** - Moderate ⚪
                - **101-150** - Unhealthy (Sensitive) 🟡
                - **151-200** - Unhealthy 🟠
                - **201-300** - Very Unhealthy 🔴
                - **301-500** - Hazardous ☠️
                """)
            
            st.divider()
            
            # Detailed comparison table
            st.markdown("**Conversion Reference**")
            
            scale_comparison = pd.DataFrame({
                'CAQI': ['1', '2', '3', '4', '5'],
                'EPA': ['0-50', '51-100', '101-150', '151-200', '201-300+'],
                'PM2.5 (µg/m³)': ['0-12', '12-35', '35-55', '55-150', '150+'],
                'Category': ['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor']
            })
            
            st.dataframe(
                scale_comparison,
                hide_index=True,
                use_container_width=True
            )
            
            st.caption("📌 This project uses OpenWeather CAQI (1-5), aligned with European/WHO standards")
    
    # ========== MAIN CONTENT ==========
    
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
    st.caption("Models loaded from MongoDB GridFS | Updates every hour")


if __name__ == "__main__":
    main()