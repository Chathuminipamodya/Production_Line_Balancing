# Smart Production Line Balancing and Workforce Planning with Fatigue Forcasting

## Overview

This project focuses on improving production line workforce planning by combining absenteeism prediction, employee fatigue forecasting, skill-based replacement selection, and quality classification.

The system aims to support production planning decisions by considering workforce availability, employee fatigue levels, operational skills, and quality-related information.

## Project Objectives

- Forecast employee absenteeism to support workforce planning.
- Predict employee fatigue using time-series forecasting techniques.
- Rank potential replacement employees based on skill similarity and predicted fatigue.
- Classify production quality using machine learning.
- Support more informed workforce allocation and production line balancing decisions.

## Project Modules

### 1. Absentee Prediction

This module predicts the expected number of employee absences using historical absenteeism data and time-based features.

Machine learning models explored include:

- Poisson Regression
- Random Forest Regression
- Gradient Boosting
- XGBoost

Model performance is evaluated using metrics such as MAE, RMSE, R², and MAPE.

### 2. Employee Fatigue Forecasting

This module analyzes fatigue patterns over time and forecasts future fatigue levels.

Forecasting approaches include:

- ARIMA
- Long Short-Term Memory (LSTM) neural networks

The results can help identify increasing fatigue trends and support workforce planning decisions.

### 3. Skill-Based Employee Replacement

This module uses employee skill information to rank potential replacements when an employee is unavailable.

The ranking approach combines skill mismatch with a fatigue-related penalty. Candidates with lower adjusted scores are ranked as better potential replacements, subject to the available skills and candidate pool.

### 4. Production Quality Classification

This module applies machine learning to classify production quality using the features available in the dataset.

## Technologies Used

- **Programming:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost
- **Time-series forecasting:** Statsmodels, TensorFlow/Keras
- **Visualization:** Matplotlib, Seaborn
- **Development environment:** Google Colab and Jupyter Notebook

## Repository Structure

```text
Smart-Production-Line-Balancing/
├── notebooks/
│   ├── Absentee_Prediction.ipynb
│   └── Production_Line_Fatigue_Analysis.ipynb
├── data/
├── models/
├── outputs/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

The `data/`, `models/`, `outputs/`, and `src/` directories contain datasets, trained models, generated results, and reusable source code, respectively, where available.

## How to Run

1. Clone or download this repository.
2. Install the required Python packages listed in `requirements.txt`.
3. Open the relevant notebook in Google Colab or Jupyter Notebook.
4. Ensure the required datasets are available in the expected locations.
5. Run the notebook cells in order to perform data preparation, model training, evaluation, and prediction.

Some notebooks may require additional configuration or dataset uploads before execution.

## Evaluation

The models are evaluated using appropriate regression metrics, including:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination
- **MAPE:** Mean Absolute Percentage Error

Lower MAE, RMSE, and MAPE generally indicate lower prediction errors, while a higher R² generally indicates a better fit relative to a mean-prediction baseline.

## Limitations

- Forecasting quality depends on the availability and quality of historical data.
- Unexpected absences and production disruptions may not be captured by historical patterns.
- Replacement rankings depend on the available skill matrix and the scoring rules.
- Predictions are decision-support estimates and should not be treated as guaranteed outcomes.

## Future Improvements

- Integrate absenteeism and fatigue forecasts into a unified workforce planning workflow.
- Incorporate machine compatibility, workload limits, and production-line constraints.
- Improve model validation using consistent time-based evaluation procedures.
- Develop a dashboard for monitoring predictions and replacement recommendations.

## Data Privacy

Datasets may contain sensitive employee or operational information. Do not publish personally identifiable or confidential data without authorization. Use anonymized or approved sample data when sharing this repository publicly.

## Disclaimer

This project is intended for research, learning, and decision-support purposes. Model predictions should be validated before being used in real production environments.
