import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

"""## Step 1: Load the Dataset

If running in **Google Colab**, upload the CSV first.
"""

from google.colab import files
uploaded = files.upload()

FILE_NAME = "Absenteeism_150000.csv"   # change if your filename differs

df = pd.read_csv(FILE_NAME, parse_dates=['Date'])
print("Shape:", df.shape)
df.head()

"""## Step 2: Inspect & Clean

We only need `Date` and `Absent` for the count-prediction target, but we keep the
raw dataframe intact for reference. Basic cleaning: correct dtypes, drop exact duplicate
rows, and validate `Absent` is binary (0/1).
"""

print(df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nAbsent value counts:\n", df['Absent'].value_counts())

# Drop exact duplicate rows (safe cleaning step, does not affect the date-level aggregation logic)
df = df.drop_duplicates().reset_index(drop=True)

# Ensure Absent is strictly 0/1 integer
df['Absent'] = df['Absent'].astype(int)
df = df[df['Absent'].isin([0, 1])]

print("Cleaned shape:", df.shape)

"""## Step 3: Aggregate to Daily Absentee Counts

This is the core transformation: turn the row-per-employee-per-day dataset into a
**row-per-date** series with the target `AbsentCount` = number of employees absent
that day.

We also **reindex to a continuous daily calendar** (filling any missing dates with 0)
so lag/rolling features are computed correctly — this is essential for time-series
accuracy.
"""

daily = df.groupby('Date')['Absent'].sum().reset_index()
daily.columns = ['Date', 'AbsentCount']

# Reindex to a full continuous daily calendar (fills any gaps with 0 absentees)
full_range = pd.date_range(daily['Date'].min(), daily['Date'].max(), freq='D')
daily = daily.set_index('Date').reindex(full_range).fillna(0).rename_axis('Date').reset_index()
daily['AbsentCount'] = daily['AbsentCount'].astype(int)

print("Date range:", daily['Date'].min().date(), "->", daily['Date'].max().date())
print("Total days:", len(daily))
daily.describe()

# Quick visual sanity check
plt.figure(figsize=(14,4))
plt.plot(daily['Date'], daily['AbsentCount'], linewidth=0.8)
plt.title('Daily Absentee Count Over Time')
plt.xlabel('Date'); plt.ylabel('Absent Count')
plt.tight_layout()
plt.show()

# Weekly pattern check — this is real, exploitable signal
daily['DOW_name'] = daily['Date'].dt.day_name()
order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
plt.figure(figsize=(8,4))
sns.barplot(data=daily, x='DOW_name', y='AbsentCount', order=order, errorbar=None)
plt.title('Average Absent Count by Day of Week')
plt.tight_layout()
plt.show()
daily = daily.drop(columns=['DOW_name'])

"""## Step 4: Feature Engineering

We build three feature families:

- **Calendar features**: day-of-week, month, quarter, day-of-year, weekend flag,
  cyclical sin/cos encodings (so Dec 31 → Jan 1 is treated as "close" numerically).
- **Lag features**: absent count on previous 1, 2, 3, 7, 14, 30 days — captures
  short-term momentum and weekly seasonality.
- **Rolling statistics**: rolling mean/std over 7, 14, 30-day windows (computed on
  **shifted** data so today's value is never leaked into its own features).
"""

def build_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d['DayOfWeek']  = d['Date'].dt.dayofweek        # 0=Mon .. 6=Sun
    d['Day']        = d['Date'].dt.day
    d['Month']      = d['Date'].dt.month
    d['Quarter']    = d['Date'].dt.quarter
    d['Year']       = d['Date'].dt.year
    d['DayOfYear']  = d['Date'].dt.dayofyear
    d['WeekOfYear'] = d['Date'].dt.isocalendar().week.astype(int)
    d['IsWeekend']  = (d['DayOfWeek'] >= 5).astype(int)
    d['IsMonthStart'] = d['Date'].dt.is_month_start.astype(int)
    d['IsMonthEnd']   = d['Date'].dt.is_month_end.astype(int)

    # cyclical encodings
    d['dow_sin']   = np.sin(2*np.pi*d['DayOfWeek']/7)
    d['dow_cos']   = np.cos(2*np.pi*d['DayOfWeek']/7)
    d['month_sin'] = np.sin(2*np.pi*d['Month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['Month']/12)

    # lag features (past absentee counts)
    for lag in [1, 2, 3, 7, 14, 30]:
        d[f'lag_{lag}'] = d['AbsentCount'].shift(lag)

    # rolling statistics (shifted by 1 so "today" is excluded)
    for w in [7, 14, 30]:
        d[f'rollmean_{w}'] = d['AbsentCount'].shift(1).rolling(w).mean()
        d[f'rollstd_{w}']  = d['AbsentCount'].shift(1).rolling(w).std()

    return d

featured = build_features(daily)
featured = featured.dropna().reset_index(drop=True)   # drop initial rows without full lag history
print("Rows available after feature engineering:", featured.shape)
featured.head()

"""## Step 5: Time-Based Train/Test Split

**Never shuffle time-series data.** We hold out the most recent 15% of days as the
test set to simulate real forecasting on unseen future dates.
"""

FEATURE_COLS = [c for c in featured.columns if c not in ['Date', 'AbsentCount']]
TARGET_COL = 'AbsentCount'

split_idx = int(len(featured) * 0.85)
train_df = featured.iloc[:split_idx]
test_df  = featured.iloc[split_idx:]

X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
X_test,  y_test  = test_df[FEATURE_COLS],  test_df[TARGET_COL]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Train period:", train_df['Date'].min().date(), "->", train_df['Date'].max().date())
print("Test period :", test_df['Date'].min().date(), "->", test_df['Date'].max().date())

"""## Step 6: Baseline & Model Comparison

We evaluate a **day-of-week average baseline** (a naive but strong benchmark for count
data like this) against four models. Because absentee counts are **non-negative small
integers**, Poisson-objective models (`PoissonRegressor`, `XGBoost` with
`count:poisson`) are theoretically the best-suited family — we include them
alongside standard regressors for comparison.
"""

def evaluate(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1, y_true))) * 100
    print(f"{name:22s} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f} | MAPE: {mape:.1f}%")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

results = []

# Baseline: predict the historical mean for that day-of-week
dow_mean = train_df.groupby('DayOfWeek')[TARGET_COL].mean()
baseline_pred = test_df['DayOfWeek'].map(dow_mean)
results.append(evaluate(y_test, baseline_pred, "Baseline (DOW mean)"))

# Poisson Regression
poisson = PoissonRegressor(alpha=0.1, max_iter=2000)
poisson.fit(X_train, y_train)
results.append(evaluate(y_test, poisson.predict(X_test), "Poisson Regression"))

# Random Forest
rf = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results.append(evaluate(y_test, rf.predict(X_test), "Random Forest"))

# Gradient Boosting
gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, random_state=42)
gbr.fit(X_train, y_train)
results.append(evaluate(y_test, gbr.predict(X_test), "Gradient Boosting"))

# XGBoost - Poisson objective (best suited for count data)
xgb_model = xgb.XGBRegressor(
    objective='count:poisson', n_estimators=400, max_depth=4,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_model.fit(X_train, y_train)
results.append(evaluate(y_test, xgb_model.predict(X_test), "XGBoost (Poisson)"))

results_df = pd.DataFrame(results).sort_values('MAE')
results_df

"""## Step 7: Hyperparameter Tuning (Best Model)

We take the best-performing model family (XGBoost, Poisson objective) and tune it with
`RandomizedSearchCV` using `TimeSeriesSplit` cross-validation — this respects the
temporal order during CV (each fold only trains on the past and validates on the
"future" relative to that fold), which is essential for realistic accuracy estimates.
"""

param_dist = {
    'n_estimators':      [200, 300, 400, 600, 800],
    'max_depth':         [3, 4, 5, 6, 8],
    'learning_rate':     [0.01, 0.02, 0.03, 0.05, 0.08],
    'subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight':  [1, 3, 5, 7],
    'reg_alpha':         [0, 0.01, 0.1, 1],
    'reg_lambda':        [0.5, 1, 2, 5],
}

tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=xgb.XGBRegressor(objective='count:poisson', random_state=42),
    param_distributions=param_dist,
    n_iter=60,
    scoring='neg_mean_absolute_error',
    cv=tscv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)
print("Best CV MAE:", -search.best_score_)
print("Best params:", search.best_params_)

best_model = search.best_estimator_

"""## Step 8: Final Model Evaluation on Held-Out Test Set"""

final_pred = best_model.predict(X_test)
final_pred_rounded = np.clip(np.round(final_pred), 0, None)  # counts must be whole, non-negative

final_metrics = evaluate(y_test, final_pred, "Tuned XGBoost (raw)")
final_metrics_r = evaluate(y_test, final_pred_rounded, "Tuned XGBoost (rounded)")

plt.figure(figsize=(14,5))
plt.plot(test_df['Date'], y_test.values, label='Actual', marker='o', markersize=3)
plt.plot(test_df['Date'], final_pred_rounded, label='Predicted', marker='x', markersize=3)
plt.title('Actual vs Predicted Daily Absent Count (Test Period)')
plt.xlabel('Date'); plt.ylabel('Absent Count')
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance
importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
plt.figure(figsize=(8,8))
sns.barplot(x=importances.values[:15], y=importances.index[:15])
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()
importances.head(15)

"""## Step 9: Refit on Full Dataset (Train + Test)

For the deployed model, retrain the tuned model on **all available history** so
future predictions use the most complete data.
"""

X_full, y_full = featured[FEATURE_COLS], featured[TARGET_COL]

production_model = xgb.XGBRegressor(**search.best_params_, objective='count:poisson', random_state=42)
production_model.fit(X_full, y_full)

joblib.dump(production_model, 'absentee_forecast_model.pkl')
joblib.dump(FEATURE_COLS, 'absentee_feature_cols.pkl')
print("Model saved: absentee_forecast_model.pkl")

"""## Step 10: Predict Absentee Count for Any Given Date

For a date **within/near known history**, we can compute real lag/rolling features
directly. For a date **further in the future**, lag features are not directly
available — so we forecast **recursively**: predict day by day, feeding each day's
prediction back in as a "known" value for the next day's lag features.
"""

def predict_absentees(target_date, history_df=daily, model=production_model, feature_cols=FEATURE_COLS):
    """
    Predict the number of absentees for `target_date` (str 'YYYY-MM-DD' or Timestamp).
    Uses actual history where available, and recursive forecasting for future dates
    beyond the known data range.
    """
    target_date = pd.to_datetime(target_date)
    hist = history_df[['Date', 'AbsentCount']].copy().sort_values('Date').reset_index(drop=True)
    last_known_date = hist['Date'].max()

    if target_date <= last_known_date:
        # Date within history: build features directly from known data
        feat = build_features(hist)
        row = feat[feat['Date'] == target_date]
        if row.empty:
            raise ValueError("Target date not found and earlier than the earliest usable feature row "
                              "(needs >=30 days of prior history).")
        pred = model.predict(row[feature_cols])[0]
        return max(0, round(float(pred)))

    # Future date beyond history: forecast day by day
    working = hist.copy()
    current_date = last_known_date + pd.Timedelta(days=1)
    prediction = None
    while current_date <= target_date:
        feat = build_features(working)
        row = feat[feat['Date'] == working['Date'].max()]  # last row's features not used directly
        # Append a placeholder row for current_date, then build its features from 'working' history
        temp = pd.concat([working, pd.DataFrame({'Date':[current_date], 'AbsentCount':[np.nan]})], ignore_index=True)
        feat_temp = build_features(temp)
        feat_row = feat_temp[feat_temp['Date'] == current_date][feature_cols]
        pred = model.predict(feat_row)[0]
        pred = max(0, round(float(pred)))
        working = pd.concat([working, pd.DataFrame({'Date':[current_date], 'AbsentCount':[pred]})], ignore_index=True)
        prediction = pred
        current_date += pd.Timedelta(days=1)

    return prediction

# Example usage
print("Predicted absentees for a date within history (2025-06-10):",
      predict_absentees('2025-06-10'))

print("Predicted absentees for a future date (2026-06-15):",
      predict_absentees('2026-06-15'))

import os
import zipfile
from google.colab import files

files_to_zip = [
    "absentee_forecast_model.pkl",
    "absentee_feature_cols.pkl"
]

with zipfile.ZipFile("absentee_prediction_files.zip", "w") as z:
    for file in files_to_zip:
        if os.path.exists(file):
            z.write(file)

files.download("absentee_prediction_files.zip")

import os

for filename in [
    "absentee_forecast_model.pkl",
    "absentee_feature_cols.pkl"
]:
    if os.path.exists(filename):
        print(f"FOUND: {filename}")
    else:
        print(f"NOT FOUND: {filename}")

from google.colab import files

files.download("absentee_forecast_model.pkl")

files.download("absentee_feature_cols.pkl")

from google.colab import drive
drive.mount('/content/drive')