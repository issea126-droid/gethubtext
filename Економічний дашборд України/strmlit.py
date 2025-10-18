import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

# Налаштування сторінки
st.set_page_config(layout="wide", page_title="Економічний дашборд України")

# --- Функція для завантаження даних з World Bank ---
@st.cache_data(ttl=3600)
def fetch_wb_indicator(country="UKR", indicator="NY.GDP.MKTP.CD", per_page=1000):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page={per_page}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return pd.DataFrame(columns=["date", "value"])
    data = r.json()
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame(columns=["date", "value"])
    records = data[1]
    rows = []
    for rec in records:
        year = rec.get("date")
        val = rec.get("value")
        if val is not None:
            rows.append({"date": int(year), "value": float(val)})
    df = pd.DataFrame(rows).sort_values("date")
    return df

# --- Економічні показники ---
INDICATORS = {
    "ВВП (current US$)": "NY.GDP.MKTP.CD",
    "Інфляція (annual %)": "FP.CPI.TOTL.ZG",
    "Безробіття (%)": "SL.UEM.TOTL.ZS"
}

# --- Заголовок ---
st.title("📊 Економічний дашборд України")
st.write("Дані з World Bank API: інфляція, ВВП, безробіття. Побудова графіків, кореляцій, прогноз на 6 періодів.")

# --- Панель керування ---
with st.sidebar:
    years_back = st.slider("Кількість років для перегляду", 5, 40, 20)
    forecast_periods = st.slider("Кількість прогнозних періодів (років)", 1, 10, 6)
    indicator_to_forecast = st.selectbox("Показник для прогнозу", list(INDICATORS.keys()))
    show_corr = st.checkbox("Показати кореляцію", True)

# --- Завантаження даних ---
dfs = {}
min_year = datetime.now().year - years_back
for name, code in INDICATORS.items():
    df = fetch_wb_indicator(indicator=code)
    df = df[df["date"] >= min_year]
    dfs[name] = df.reset_index(drop=True)

# --- Об’єднання у таблицю ---
years = sorted({y for df in dfs.values() for y in df["date"]})
combined = pd.DataFrame({"Рік": years})
for name, df in dfs.items():
    combined = combined.merge(df.rename(columns={"value": name, "date": "Рік"}), on="Рік", how="left")

st.subheader("📋 Таблиця економічних даних")
st.dataframe(combined)

# --- Побудова графіків ---
st.subheader("📈 Графіки показників")
chosen = st.multiselect("Оберіть показники", list(INDICATORS.keys()), default=list(INDICATORS.keys()))
if chosen:
    fig, ax = plt.subplots()
    for c in chosen:
        ax.plot(combined["Рік"], combined[c], marker="o", label=c)
    ax.legend()
    ax.set_xlabel("Рік")
    ax.set_ylabel("Значення")
    ax.set_title("Динаміка економічних показників України")
    st.pyplot(fig)
else:
    st.info("Оберіть хоча б один показник для побудови графіка.")

# --- Кореляція ---
if show_corr:
    st.subheader("📊 Кореляції між показниками")
    corr = combined.drop(columns=["Рік"]).corr(method="pearson")
    st.dataframe(corr.style.format("{:.3f}"))

# --- Прогноз ---
st.subheader("🔮 Прогноз обраного показника")
df = dfs[indicator_to_forecast].dropna()
if len(df) < 5:
    st.warning("Недостатньо даних для прогнозу.")
else:
    y = df["value"].values
    years = df["date"].values

    try:
        model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,0,0,0))
        res = model.fit(disp=False)
        forecast = res.get_forecast(steps=forecast_periods)
        pred = forecast.predicted_mean
        ci = forecast.conf_int()

        future_years = np.arange(years[-1] + 1, years[-1] + 1 + forecast_periods)

        fig, ax = plt.subplots()
        ax.plot(years, y, marker="o", label="Історичні дані")
        ax.plot(future_years, pred, marker="o", color="red", label="Прогноз")
        ax.fill_between(future_years, ci.iloc[:, 0], ci.iloc[:, 1], color="pink", alpha=0.3)
        ax.legend()
        ax.set_xlabel("Рік")
        ax.set_title(f"Прогноз: {indicator_to_forecast}")
        st.pyplot(fig)

        forecast_df = pd.DataFrame({
            "Рік": future_years,
            "Прогноз": pred,
            "Нижня межа": ci.iloc[:, 0],
            "Верхня межа": ci.iloc[:, 1]
        })
        st.dataframe(forecast_df.style.format("{:.2f}"))
    except Exception as e:
        st.error(f"Помилка при прогнозуванні: {e}")

# --- Кнопка для збереження ---
st.markdown("---")
st.download_button("⬇️ Завантажити CSV", combined.to_csv(index=False), "econ_data.csv")
