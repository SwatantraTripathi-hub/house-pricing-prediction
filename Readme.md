# 🏡 Real Estate Price Prediction Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange.svg)](https://scikit-learn.org/)
[![R² Score](https://img.shields.io/badge/R²%20Accuracy-0.96-success.svg)](https://github.com/)

An end-to-end machine learning web application built to predict property prices dynamically. The core model achieves an exceptional **$R^2$ accuracy score of 0.96**, backed by an interactive, real-time frontend dashboard powered by **Streamlit**.

---

## ✨ Key Features
- **High-Accuracy Predictions**: Powered by an optimized regression pipeline achieving a **0.96 $R^2$ score**.
- **Interactive UI**: Real-time property price estimation using dynamic Streamlit sliders and input widgets.
- **Model Serialization**: Uses `joblib`/`pickle` for instant loading of trained models and feature scalers.
- **Production-Ready Structure**: Clean separation of model training, data generation, and web deployment layers.

---

## 🛠️ Tech Stack
* **Language**: Python
* **Machine Learning**: Scikit-Learn, Pandas, NumPy
* **Frontend & Web App**: Streamlit
* **Model Persistence**: Joblib / Pickle
* **Development Tools**: VS Code, GitHub Copilot

---

## 📁 Project Directory Structure
```text
House_Price_Prediction/
│
├── .vscode/                 # VS Code workspace settings
├── templates/               # HTML templates (legacy/optional)
├── app.py                   # Main Streamlit web application
├── train_model.py           # ML training script & evaluation pipeline
├── generate_data.py         # Synthetic dataset generator script
├── house_data_1000.csv      # Housing dataset (1000+ records)
├── house_model.pkl          # Serialized machine learning model & scaler
└── requirements.txt         # Project Python dependencies