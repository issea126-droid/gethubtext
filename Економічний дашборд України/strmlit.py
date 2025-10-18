import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from datetime import datetime

st.set_page_config(layout="wide", page_title="Економічний дашборд України")

# --- Helpers: загрузка данных World Bank ---
@st.cache_data(ttl=3600)
def fetch_wb_indicator(country="UKR", indicator="NY.GDP.MKTP.CD", per_page=1000):
    """
    Возвращает DataFrame с колонками ['date','value'] для индикатора WB.
    indicator: код WB (например 'NY.GDP.MKTP.CD' для GDP (current US$))
    """
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page={per_page}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame(columns=["date","value"])
    records = data[1]
    rows = []
    for rec in records:
        year = rec.get("date")
        val = rec.get("value")
        if val is None:
            continue
        rows.append({"date": int(year), "value": float(val)})
    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df

# Параметры индикаторов
INDICATORS = {
    "GDP (current US$)": "NY.GDP.MKTP.CD",
    "Inflation, consumer prices (annual %)": "FP.CPI.TOTL.ZG",
    "Unemployment, total (% of total labor force)": "SL.UEM.TOTL.ZS"
}

st.title("Економічний дашборд України")
st.markdown("Дані: World Bank API — інфляція, ВВП, безробіття (річні). Інтерактивні графіки, кореляції, прогноз на 6 періодів.")

# Ліва панель: настройки
with st.sidebar:
    st.header("Налаштування")
    years_back = st.slider("Показати останніх років", min_value=5, max_value=40, value=20)
    forecast_periods = st.slider("Горизонт прогнозу (периодів — відповідає рокам)", min_value=1, max_value=12, value=6)
    show_corr = st.checkbox("Показати матрицю кореляцій", value=True)
    indicator_to_forecast = st.selectbox("Що прогнозувати?", list(INDICATORS.keys()))
    st.markdown("Примітка: World Bank дає річні дані — прогноз також річний. Для місячного прогнозу потрібні місячні джерела.")

# Load data for each indicator
dfs = {}
min_year = datetime.now().year - years_back
for name, code in INDICATORS.items():
    df = fetch_wb_indicator(indicator=code)
    # фильтруем последние years_back лет
    df = df[df["date"] >= min_year].copy()
    df = df.reset_index(drop=True)
    dfs[name] = df

# Объединяем в один DataFrame по году
all_years = sorted({y for df in dfs.values() for y in df["date"].unique()})
combined = pd.DataFrame({"date": all_years})
for name, df in dfs.items():
    combined = combined.merge(df.rename(columns={"value": name}), on="date", how="left")
combined = combined.sort_values("date").reset_index(drop=True)

st.subheader("Дані (оновлено з World Bank)")
st.dataframe(combined.style.format({c: "{:,.2f}" for c in combined.columns if c!="date"}))

# --- Интерактивные графики ---
st.subheader("Графіки показників")
col1, col2 = st.columns([2,1])

with col1:
    # выбор показателей для графика
    chosen = st.multiselect("Оберіть показники для графіка", list(INDICATORS.keys()), default=list(INDICATORS.keys()))
    if not chosen:
        st.info("Оберіть хоча б один показник.")
    else:
        fig = px.line(combined, x="date", y=chosen, markers=True)
        fig.update_layout(xaxis_title="Рік", yaxis_title="Значення", legend_title="Показники")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Опис")
    st.markdown("- Дані — річні. Якщо для інфляції потрібен місячний розподіл, треба підключити інші API (напр. нацбанк або статистику).")
    st.markdown("- Можна вибрати показники та побудувати графік.")

# --- Кореляції ---
if show_corr:
    st.subheader("Кореляції між показниками (Пірсон)")
    corr_df = combined.drop(columns=["date"]).corr(method="pearson")
    st.write("Матриця кореляцій:")
    st.dataframe(corr_df.style.format("{:.3f}"))
    st.markdown("Коли є пропуски, вони ігноруються при обчисленні кореляції парно.")

    # также вывести пары с наибольшими корреляциями
    cols = corr_df.columns.tolist()
    pairs = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a,b = cols[i], cols[j]
            val = corr_df.loc[a,b]
            pairs.append((a,b,val))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    st.markdown("Найсильніші кореляції (за абсолютною величиною):")
    for a,b,val in pairs_sorted:
        st.write(f"{a} ↔ {b}: {val:.3f}")

# --- Прогноз ---
st.subheader("Прогноз обраного показника")
series_df = combined[["date", indicator_to_forecast]].dropna().copy()
if series_df.empty or len(series_df) < 5:
    st.warning("Недостатньо даних для прогнозу (потрібно мінімум ~5 значень).")
else:
    series_df = series_df.set_index("date")
    y = series_df[indicator_to_forecast].astype(float)

    st.write(f"Вихідні дані для прогнозу ({len(y)} точок): від {y.index.min()} до {y.index.max()}")

    # простая модель SARIMAX(1,1,1)
    try:
        model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=forecast_periods)
        pred_mean = pred.predicted_mean
        pred_ci = pred.conf_int()

        # собрать датафрейм с прогнозом
        last_year = int(y.index.max())
        future_years = [last_year + i for i in range(1, forecast_periods+1)]
        forecast_df = pd.DataFrame({
            "date": future_years,
            "forecast": pred_mean.values,
            "lower": pred_ci.iloc[:,0].values,
            "upper": pred_ci.iloc[:,1].values
        })

        plot_df = pd.concat([
            y.reset_index().rename(columns={indicator_to_forecast:"value"}),
            forecast_df.rename(columns={"forecast":"value"})[["date","value"]]
        ], ignore_index=True)

        figf = px.line(plot_df, x="date", y="value", markers=True, title=f"Прогноз {indicator_to_forecast}")
        # add CI as ribbon
        figf.add_traces(px.line(forecast_df, x="date", y="lower").data + px.line(forecast_df, x="date", y="upper").data)
        # add shaded area manually
        figf.add_traces([
            px.scatter(forecast_df, x="date", y="forecast").data[0]
        ])
        # simpler: draw forecast line and CI band using add_traces
        figf.add_traces([
            px.line(forecast_df, x="date", y="forecast").data[0]
        ])
        # render
        st.plotly_chart(figf, use_container_width=True)

        st.markdown("Прогнозна таблиця:")
        st.dataframe(forecast_df.style.format({ "forecast":"{:.2f}", "lower":"{:.2f}", "upper":"{:.2f}"}))
    except Exception as e:
        st.error(f"Помилка при будуванні моделі: {e}")

# --- Экспорт данных ---
st.markdown("---")
st.markdown("### Завантажити дані")
csv = combined.to_csv(index=False)
st.download_button("Скачати CSV з усіма даними", csv, file_name="ukr_econ_data.csv", mime="text/csv")

st.markdown("### Поради для покращення")
st.markdown("""
- Якщо треба місячні/щоквартальні дані (щоб прогноз робити на місяці), підключи API Нацбанку/Держстату або TradingEconomics.  
- Можна додати мультимодельний прогноз (VAR) для взаємозалежних показників.  
- Для більш надійного прогнозу використовуй добір параметрів (grid search або pmdarima.auto_arima).
""")
