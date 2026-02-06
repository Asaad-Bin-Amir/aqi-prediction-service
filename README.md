AQI Prediction Service

Real-time Air Quality Index forecasting system for Karachi, Pakistan using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production-success)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Data Source](#data-source)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project provides **24h, 48h, and 72h AQI forecasts** using real-time air quality and weather data.

**Key Highlights:**
- ✅ Real-time data collection (hourly via OpenWeather API)
- ✅ Multi-pollutant monitoring (PM2.5, PM10, O3, CO, NO2, SO2)
- ✅ Machine Learning forecasting (XGBoost, Random Forest, Gradient Boosting)
- ✅ Interactive web dashboard (Streamlit)
- ✅ RESTful API (FastAPI)
- ✅ Automated training pipeline (weekly retraining)
- ✅ Production-ready deployment

---

## ⚡ Features

### Data Collection
- **Hourly automated collection** via GitHub Actions
- **Multi-pollutant data**: PM2.5, PM10, O3, CO, NO2, SO2, NH3
- **Comprehensive weather**: Temperature, humidity, pressure, wind, clouds
- **Duplicate detection**: Prevents redundant data storage
- **MongoDB storage**: Scalable cloud database

### Machine Learning
- **Multiple algorithms**: XGBoost, Gradient Boosting, Random Forest, Ridge
- **Automatic model selection**: Best model chosen based on validation MAE
- **Feature engineering**: 50+ features (lags, rolling stats, interactions)
- **Time-series validation**: Proper temporal split (80/20)
- **Weekly retraining**: Adapts to changing patterns

### Visualization
- **Interactive dashboard**: Real-time AQI, forecasts, historical trends
- **Color-coded indicators**: Intuitive health category visualization
- **Trend analysis**: 7-day historical charts
- **Multi-pollutant tracking**: PM2.5, PM10, O3 trends

### API
- **RESTful endpoints**: Current AQI, forecasts, historical data
- **JSON responses**: Easy integration with other systems
- **Health checks**: Monitor system status
- **Documentation**: Auto-generated Swagger/OpenAPI docs

---

## 🏗️ Architecture

```
┌─────────────┐
│ OpenWeather │  Hourly data collection
│     API     │  (PM2.5, PM10, O3, Weather)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   MongoDB   │  Feature store
│    Atlas    │  (Cloud database)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Feature   │  Lag, rolling, time features
│ Engineering │  (50+ features)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  ML Models  │  XGBoost, GradientBoosting
│  Training   │  (24h, 48h, 72h horizons)
└──────┬──────┘
       │
       ├─────────────┐
       ▼             ▼
┌─────────────┐ ┌─────────────┐
│  Dashboard  │ │  REST API   │
│ (Streamlit) │ │  (FastAPI)  │
└─────────────┘ └─────────────┘
```

---

## 📊 Data Source

### OpenWeather API (Single Source Solution)

**Why OpenWeather:**
- Industry-standard air quality and weather data
- 99.9% uptime, highly reliable
- Free tier: 1,000,000 API calls/month
- Global coverage (works for any location)

**Data Collected (Hourly):**

**Air Quality:**
- AQI (1-5 scale, European Index)
- PM2.5 concentration (µg/m³)
- PM10 concentration (µg/m³)
- O3, CO, NO2, SO2, NH3 (µg/m³)

**Weather:**
- Temperature, feels-like (°C)
- Humidity (%)
- Atmospheric pressure (hPa)
- Wind speed, direction (m/s, degrees)
- Cloud cover (%)
- Visibility (meters)

**AQI Scale:** 1-5 (OpenWeather European Index)
- 1 = Good
- 2 = Fair
- 3 = Moderate
- 4 = Poor
- 5 = Very Poor

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- MongoDB Atlas account (free tier)
- OpenWeather API key (free tier)

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/aqi-prediction-service.git
cd aqi-prediction-service
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

Create `.env` file:

```env
# MongoDB
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/

# OpenWeather API
OPENWEATHERMAP_API_KEY=your_api_key_here
```

### Step 5: Test Connection

```bash
python src/data_collection.py
```

---

## 💻 Usage

### Collect Data

```bash
# Manual collection
python src/data_collection.py

# Automated (runs hourly via GitHub Actions)
```

### Train Models

```bash
# Train all forecast horizons (24h, 48h, 72h)
python src/training_pipeline.py

# Automated training (runs weekly via GitHub Actions)
python src/train_models_automated.py
```

### Run Dashboard

```bash
streamlit run src/dashboard.py
```

Access at: `http://localhost:8501`

### Run API

```bash
uvicorn src.forecast_api:app --reload
```

Access at: `http://localhost:8000`
API docs: `http://localhost:8000/docs`

---

## 📡 API Documentation

### Endpoints

**GET `/current`** - Current air quality
```json
{
  "location": "Karachi",
  "aqi": 4.2,
  "aqi_category": "Poor",
  "pm2_5": 95.3,
  "temperature": 28.5,
  "timestamp": "2026-02-06T12:00:00Z"
}
```

**GET `/forecast/{horizon}`** - Forecast for 24h, 48h, or 72h
```json
{
  "horizon": "24h",
  "predicted_aqi": 3.8,
  "aqi_category": "Moderate",
  "forecast_time": "2026-02-07T12:00:00Z",
  "model_name": "XGBOOST",
  "mae": 0.52,
  "r2": 0.78
}
```

**GET `/forecast/all`** - All forecasts

**GET `/history?days=7`** - Historical data

**GET `/health`** - System health check

---

## 📈 Model Performance

### Expected Metrics (after 2 weeks of data collection)

| Horizon | Model | MAE | RMSE | R² |
|---------|-------|-----|------|----|
| 24h | XGBoost | 0.45-0.65 | 0.60-0.85 | 0.75-0.85 |
| 48h | Gradient Boosting | 0.55-0.75 | 0.70-0.95 | 0.68-0.80 |
| 72h | Gradient Boosting | 0.65-0.85 | 0.80-1.05 | 0.60-0.75 |

**Note:** MAE on 1-5 scale (e.g., MAE=0.5 means ±0.5 AQI units)

### Current Status (Feb 6, 2026)

**Data Collection:** Active (started Feb 2, 2026)
- Current records: Collecting hourly
- Target: 336 records by Feb 16 (2 weeks)

**Models:** Will be trained after sufficient data (100+ records)

---

## 📁 Project Structure

```
aqi-prediction-service/
├── .github/
│   └── workflows/
│       ├── data_collection.yml    # Hourly data collection
│       └── model_training.yml     # Weekly model training
├── src/
│   ├── data_collection.py         # OpenWeather data collector
│   ├── feature_store.py           # MongoDB interface
│   ├── feature_engineering.py     # Feature creation
│   ├── training_pipeline.py       # Model training
│   ├── train_models_automated.py  # Automated training
│   ├── dashboard.py               # Streamlit web app
│   ├── forecast_api.py            # FastAPI REST API
│   ├── clean_duplicates.py        # Utility: Remove duplicates
│   ├── delete_bad_data.py         # Utility: Clean bad records
│   └── utils.py                   # Helper functions
├── models/                        # Trained model artifacts
├── notebooks/                     # Jupyter notebooks (EDA)
├── data/                          # Local data cache (optional)
├── .env                           # Environment variables (not in git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🌐 Deployment

### GitHub Actions (Automated)

**Data Collection:**
- Runs every hour at :05
- Triggered by: `.github/workflows/data_collection.yml`

**Model Training:**
- Runs every Sunday at 2 AM UTC
- Triggered by: `.github/workflows/model_training.yml`

### Manual Deployment

**Dashboard (Streamlit Cloud):**
1. Push to GitHub
2. Connect Streamlit Cloud to repo
3. Set environment variables
4. Deploy!

**API (Railway/Render/Heroku):**
1. Add `Procfile`: `web: uvicorn src.forecast_api:app --host 0.0.0.0 --port $PORT`
2. Connect platform to GitHub
3. Set environment variables
4. Deploy!

---

## 🔬 Technical Highlights

### Data Quality Measures
- **Duplicate detection**: Skips unchanged API responses
- **Outlier handling**: Validates data ranges
- **Missing value strategy**: Forward fill for gaps < 3 hours

### Feature Engineering (50+ features)
- **Lag features**: 1h, 3h, 6h, 12h, 24h lags
- **Rolling statistics**: Mean, std, min, max (6h, 12h, 24h windows)
- **Time features**: Hour, day of week, weekend indicator, rush hours
- **Cyclical encoding**: Sin/cos transforms for hour
- **Interactions**: Temperature×humidity, wind×pressure, PM2.5×humidity

### Model Selection
- **Automatic**: Best model chosen by validation MAE
- **Candidates**: XGBoost, Gradient Boosting, Random Forest, Ridge
- **Cross-validation**: Time-series aware split

### Production Best Practices
- ✅ Environment-based configuration
- ✅ Comprehensive logging
- ✅ Error handling and retries
- ✅ Data validation gates
- ✅ Model versioning
- ✅ Automated testing (GitHub Actions)

---

## 🐛 Challenges & Solutions

### Challenge 1: AQICN Station Outages
**Problem:** Karachi US Consulate station offline since March 2025

**Solution:** Switched to OpenWeather (more reliable, global coverage)

### Challenge 2: API Caching
**Problem:** 90% duplicate records from cached API responses

**Solution:** Duplicate detection logic (compares PM2.5 & AQI before saving)

### Challenge 3: AQI Scale Confusion
**Problem:** OpenWeather 1-5 vs EPA 0-500 scale

**Solution:** Instructor approved 1-5 scale (simpler, suitable for ML)

### Challenge 4: Multi-Pollutant AQI
**Problem:** O3 and PM2.5 both causing AQI=5

**Solution:** Document multi-pollutant problem (demonstrates complexity)

---

## 📚 References

- OpenWeather API: https://openweathermap.org/api/air-pollution
- European AQI: https://www.eea.europa.eu/themes/air/air-quality-index
- WHO Air Quality Guidelines: https://www.who.int/news-room/feature-stories/detail/what-are-the-who-air-quality-guidelines

---

## 👤 Author

**Your Name**
- University: [Your University]
- Program: [Your Program]
- Email: [Your Email]
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- OpenWeather for free API access
- MongoDB Atlas for cloud database
- GitHub Actions for automation
- Streamlit for rapid dashboard development

---

**Last Updated:** February 6, 2026