"""
AQI Prediction Dashboard - MINIMAL WORKING VERSION
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AQI Prediction Service",
    page_icon="🌍",
    layout="wide"
)

def main():
    st.title("🌍 AQI Prediction Service")
    st.caption("Real-time Air Quality Index forecasting")
    
    # Try to load data
    try:
        from feature_store import AQIFeatureStore
        
        with AQIFeatureStore() as fs:
            latest = fs.raw_features.find_one(sort=[('timestamp', -1)])
            
            if latest:
                st.success("✅ Connected to Database!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("AQI", f"{latest.get('aqi', 'N/A')}/5")
                
                with col2:
                    st.metric("PM2.5", f"{latest.get('pm2_5', 'N/A')} µg/m³")
                
                with col3:
                    st.metric("PM10", f"{latest.get('pm10', 'N/A')} µg/m³")
                
                with col4:
                    st.metric("Temperature", f"{latest.get('temperature', 'N/A')}°C")
                
                st.caption(f"Last updated: {latest.get('timestamp', 'Unknown')}")
            else:
                st.warning("No data available yet")
                
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        st.info("App is running but needs database configuration")
    
    # Model status
    st.header("🔮 Forecast Status")
    st.info("⚠️ Models need to be retrained for Python 3.13 compatibility")
    st.write("Current environment: Python 3.13")
    st.write("Models were trained with: Python 3.11")
    st.write("**Solution:** Retrain models locally with Python 3.13 and re-upload")
    
    st.markdown("---")
    st.caption("AQI Prediction Service - Minimal Mode")

if __name__ == "__main__":
    main()