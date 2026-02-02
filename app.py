import streamlit as st
import pandas as pd
import numpy as np
import io

from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    SimpleExpSmoothing,
    Holt
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -----------------------------
# UI
# -----------------------------
st.title("📈 Cane Price Forecasting")

url = st.radio(
    "Select the CSV file",
    ["cane101.csv", "cane-land-price-no.csv"]
)

# -----------------------------
# Load data
# -----------------------------
cane_df = pd.read_csv(url, header=2)
cane_df.columns = ['year', 'month', 'At']

cane_df['date'] = pd.date_range(
    start='2026-01-01',
    periods=len(cane_df),
    freq='MS'
)

cane_df = cane_df[['year', 'month', 'date', 'At']]
y = cane_df['At']

st.subheader("Raw Data")
st.dataframe(cane_df.head())


# -----------------------------
# Models (SOFT-CODE)
# -----------------------------
models = {
    'Holt-Winters': ExponentialSmoothing(
        y, trend='add', seasonal='mul', seasonal_periods=12
    ).fit(),

    'Simple ES.': SimpleExpSmoothing(y).fit(),

    'Holt': Holt(y).fit(),

    'ARIMA': ARIMA(y, order=(1, 1, 1)).fit(),

    'SARIMA': SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit()
}


# -----------------------------
# Fitted values
# -----------------------------
for name, model in models.items():
    cane_df[f'fitted_{name}'] = model.fittedvalues


# -----------------------------
# Error + Metrics (AUTO)
# -----------------------------
metrics = {}

for name in models.keys():
    error = cane_df['At'] - cane_df[f'fitted_{name}']

    MSE = np.nanmean(error ** 2)
    RMSE = np.sqrt(MSE)
    MAD = np.nanmean(np.abs(error))

    metrics[name] = {
        'MSE': MSE,
        'RMSE': RMSE,
        'MAD': MAD
    }


# -----------------------------
# Report table
# -----------------------------
report_cane_model = (
    pd.DataFrame.from_dict(metrics, orient='index')
    .reset_index()
    .rename(columns={'index': 'Model'})
)

st.subheader("📊 Model Comparison")
st.dataframe(report_cane_model)


# -----------------------------
# Best model
# -----------------------------
best_model = report_cane_model.loc[
    report_cane_model['RMSE'].idxmin(), 'Model'
]

st.success(f"🏆 BEST MODEL: {best_model}")


# -----------------------------
# Model selector
# -----------------------------
model_sort = st.radio(
    "Select the model to inspect",
    list(models.keys())
)

st.subheader(f"Model: {model_sort}")

# summary

st.code(models[model_sort].summary().as_text())

# metrics
st.write("MSE :", metrics[model_sort]['MSE'])
st.write("RMSE:", metrics[model_sort]['RMSE'])
st.write("MAD :", metrics[model_sort]['MAD'])


# -----------------------------
# Forecast
# -----------------------------
st.subheader("🔮 12-Month Forecast")
forecast_12 = models[model_sort].forecast(12)
st.write(forecast_12)
