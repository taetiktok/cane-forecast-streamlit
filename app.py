import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    SimpleExpSmoothing,
    Holt
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =====================================================
# UI
# =====================================================
st.set_page_config(page_title="Cane Price Forecast", layout="wide")
st.title("📈 Cane Price Forecasting System")

file_selected = st.radio(
    "Select the CSV file",
    ["cane101.csv", "cane-land-price-no.csv"]
)

if file_selected == "cane-land-price-no.csv":
    st.warning("⚠️ Dataset นี้อยู่ในช่วงพัฒนา")

else:
    # =====================================================
    # Load & Prepare Data
    # =====================================================
    cane_df = pd.read_csv(file_selected, header=2)
    cane_df.columns = ['year', 'month', 'At']

    cane_df['date'] = pd.date_range(
        start='2026-01-01',
        periods=len(cane_df),
        freq='MS'
    )

    cane_df = cane_df[['year', 'month', 'date', 'At']]
    y = cane_df['At']

    st.subheader("📄 Raw Data")
    st.dataframe(cane_df.head())

    # =====================================================
    # Models (SOFT-CODE)
    # =====================================================
    models = {
        'Holt-Winters': ExponentialSmoothing(
            y, trend='add', seasonal='mul', seasonal_periods=12
        ).fit(),

        'Simple ES': SimpleExpSmoothing(y).fit(),

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

    # =====================================================
    # Fitted values
    # =====================================================
    for name, model in models.items():
        cane_df[f'fitted_{name}'] = model.fittedvalues

    # =====================================================
    # Metrics (AUTO)
    # =====================================================
    metrics = {}

    for name in models.keys():
        error = cane_df['At'] - cane_df[f'fitted_{name}']

        metrics[name] = {
            'MSE': np.nanmean(error ** 2),
            'RMSE': np.sqrt(np.nanmean(error ** 2)),
            'MAD': np.nanmean(np.abs(error))
        }

    report = (
        pd.DataFrame.from_dict(metrics, orient='index')
        .reset_index()
        .rename(columns={'index': 'Model'})
    )

    st.subheader("📊 Model Comparison")
    st.dataframe(report)

    # =====================================================
    # Best Model
    # =====================================================
    best_model = report.loc[report['RMSE'].idxmin(), 'Model']
    st.success(f"🏆 BEST MODEL: {best_model}")

    # =====================================================
    # Model Selector
    # =====================================================
    model_sort = st.selectbox(
        "Select model to inspect",
        list(models.keys())
    )

    # =====================================================
    # Actual vs Fitted Plot
    # =====================================================
    st.subheader("📈 Actual vs Fitted")

    fig1, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(cane_df['date'], cane_df['At'], label="Actual", color="black")
    ax1.plot(
        cane_df['date'],
        cane_df[f'fitted_{model_sort}'],
        label="Fitted",
        linestyle="--"
    )
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # =====================================================
    # Metrics Display
    # =====================================================
    st.subheader("📐 Error Metrics")
    st.write(f"MSE  : {metrics[model_sort]['MSE']:.4f}")
    st.write(f"RMSE : {metrics[model_sort]['RMSE']:.4f}")
    st.write(f"MAD  : {metrics[model_sort]['MAD']:.4f}")

    # =====================================================
    # Model Summary
    # =====================================================
    if st.checkbox("Show model summary"):
        st.code(models[model_sort].summary().as_text())

    # =====================================================
    # Forecast
    # =====================================================
    st.subheader("🔮 Forecast (12 Months)")
    forecast_12 = models[model_sort].forecast(12)

    future_dates = pd.date_range(
        start=cane_df['date'].iloc[-1] + pd.offsets.MonthBegin(),
        periods=12,
        freq='MS'
    )

    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': forecast_12
    })

    # Forecast Plot
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(cane_df['date'], cane_df['At'], label="Actual", color="black")
    ax2.plot(forecast_df['date'], forecast_df['forecast'],
             label="Forecast", color="red")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # =====================================================
    # Download Forecast
    # =====================================================
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Forecast CSV",
        csv,
        file_name=f"forecast_{model_sort}.csv",
        mime="text/csv"
    )
