# Mental Health Work Interference Analyzer

## Overview

This project analyzes employee mental health and its interference with work performance using the OSMI 2014 Mental Health in Tech Survey dataset.

The project combines exploratory data analysis, machine learning, Power BI dashboards, and a Streamlit application to identify the personal and workplace factors associated with meaningful work interference.

## Project Objectives

- Analyze patterns of mental health-related work interference
- Identify important personal and workplace factors
- Compare Full and HR-only machine learning models
- Build an interactive Power BI dashboard
- Develop a Streamlit prediction application
- Explore a privacy-friendly model based only on workplace variables

## Dataset

- Dataset: OSMI 2014 Mental Health in Tech Survey
- Final modeling sample: 995 participants
- Countries represented: 40
- Target variable: `work_interfere`

### Target Distribution

- Meaningful work interference: 609 participants (61.21%)
- Low work interference: 386 participants (38.79%)

## Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

Two modeling scenarios were compared:

1. Full Model using personal and workplace variables
2. HR-only Model using workplace-related variables only

## Key Results

- Best-performing model: Random Forest Full Model
- ROC-AUC: 0.758
- F1 Score: 0.778
- Treatment was the strongest predictor in the Full Model
- Family history also contributed substantially
- Important HR-only factors included leave policy, company size, and mental health consequences

## Power BI Dashboard

The dashboard includes four pages:

1. Executive Dashboard
2. Employee Insights
3. Machine Learning Performance
4. Feature Importance and HR Insights

## Dashboard Preview

### Executive Dashboard

![Executive Dashboard](assets/executive_dashboard.png)

### Employee Insights

![Employee Insights](assets/employee_insights.png)

### Machine Learning Performance

![Machine Learning Performance](assets/ml_performance.png)

### Feature Importance and HR Insights

![Feature Importance and HR Insights](assets/feature_importance_hr.png)

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Power BI
- Streamlit
- GitHub

## Repository Files

- `app.py` — Streamlit application
- `model_full.pkl` — trained Full Model
- `model_hr.pkl` — trained HR-only Model
- `feature_names.json` — model feature names
- `feature_values.json` — feature input values
- `requirements.txt` — Python dependencies
- Final Jupyter notebook — complete analysis and modeling workflow
- Power BI dashboard — interactive reporting dashboard

## Ethical Considerations

The HR-only model excludes sensitive personal health variables. Although this reduces predictive performance, it provides a more privacy-conscious alternative for workplace applications.

The models are intended for educational and analytical purposes and should not be used to diagnose mental health conditions or make employment decisions.

## Author

**Ariana Moayed**
