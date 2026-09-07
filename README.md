# 📊 FinSight AI

### AI-Powered Financial Analysis & Forecasting Dashboard

<p align="center">
  <b>Analyze Financial Data • Discover Insights • Forecast Future Trends</b>
</p>

<p align="center">
  <a href="https://finsight-ai-dashboard.streamlit.app/">
    🚀 LIVE DEMO
  </a>
</p>

---

## 📌 About the Project

**FinSight AI** is an interactive financial analytics and forecasting platform built using **Python and Streamlit**.

The application helps users explore financial data, analyze key performance indicators, visualize trends, and generate forecasts using a trained machine learning model.

The project combines **Data Analytics, Machine Learning, Data Visualization, and AI** into a single interactive financial analytics platform.

---

## 🚀 Live Demo

### 👉 [🔴 Open FinSight AI — Live Application](https://finsight-ai-dashboard.streamlit.app/)

---

## ✨ Features

### 📊 Financial Data Analysis

- Financial dataset exploration
- Data cleaning and preprocessing
- Statistical summaries
- KPI analysis
- Trend identification
- Interactive data analysis

### 📈 Interactive Visualizations

- Financial trend analysis
- Interactive Plotly charts
- Historical data visualization
- Dynamic filtering
- Comparative analysis

### 🔮 Financial Forecasting

- Machine learning-based forecasting
- Historical vs predicted values
- Future trend visualization
- Trained model integration
- Model loading using Joblib

### 🤖 AI Financial Assistant

- Interactive financial chatbot
- Natural-language interaction
- AI-powered financial assistance
- Financial insights and explanations

### 🎯 Interactive Dashboard

- Streamlit-based interface
- User-friendly navigation
- Interactive controls
- Real-time analytical outputs

---

# 🏗️ Application Architecture

```text
                    ┌──────────────────────┐
                    │     User / Analyst   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Streamlit Dashboard │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
      │ Financial   │   │ Forecasting │   │ AI Chatbot  │
      │ Analytics   │   │ ML Model    │   │             │
      └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
             │                 │                 │
             ▼                 ▼                 ▼
      ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
      │ Pandas /    │   │ Scikit-Learn│   │ OpenAI API  │
      │ NumPy       │   │ + Joblib    │   │             │
      └──────┬──────┘   └──────┬──────┘   └─────────────┘
             │                 │
             └────────┬────────┘
                      ▼
             ┌──────────────────┐
             │ Financial Insights│
             └──────────────────┘
