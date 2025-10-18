import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

st.set_page_config(layout="wide", page_title="Економічний дашборд України")

st.title("Економічний дашборд України 🇺🇦")
st.markdown(
    "Коротко: завантажуємо відкриті дані (World Bank / державні API), будуємо інтерактивні графіки, кореляції та прогноз на 6 місяців."
)

# ---------------------------
# Helper: World Bank API fetch
# ---------------------------
WB_BASE = "https://api.worldbank.org/v2/country/UA/indicator/{indicator}?format=json&per_page=1000"

INDICATORS = {
    "GDP (current US$)": "NY.GDP.MKTP.CD",
    "Inflation (CPI, annual %)": "FP.CPI.TOTL.ZG",
    "Unemployment rate (% of labor force)": "SL.UEM.TOTL.ZS",
}

@st.cache_data(show_spinner=False)
def fetch_wb_series(indicator_code):
    url = WB_BASE.format(indicator=indicator_code)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    # data[1] contains observations
    if len(data) < 2 or not data[1]:
        return pd.DataFrame(columns=["date", "value"])
    records = []
    for item in data[1]:
        year = item.get('date')
        val = item.get('value')
        if val is not None:
            try:
                records.append({"date": pd.to_datetime(f"{year}-01-01"), "value": float(val)})
            except Exception:
                continue
    df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
    return df

# Sidebar controls
st.sidebar.header("Налаштування")
indicators_selected = st.sidebar.multiselect("Показники (джерело: World Bank)", list(INDICATORS.keys()), default=list(INDICATORS.keys()))
start_year = st.sidebar.number_input("Рік початку:", min_value=1960, max_value=datetime.now().year, value=2000, step=1)
show_interpolated = st.sidebar.checkbox("Інтерполювати до місяців (щоб дати прогноз, 6 місяців)", value=True)

# Fetch data
series_dfs = {}
for name in indicators_selected:
    code = INDICATORS[name]
    df = fetch_wb_series(code)
    if not df.empty:
        df = df[df['date'].dt.year >= int(start_year)].copy()
    series_dfs[name] = df

# Show data availability
st.subheader("Джерела та доступні часові ряди")
col1, col2 = st.columns([2,1])
with col1:
    for name, df in series_dfs.items():
        if df.empty:
            st.warning(f"{name}: дані не знайдені для вибраних параметрів")
        else:
            st.write(f"**{name}** — період: {df['date'].dt.year.min()} — {df['date'].dt.year.max()} ({len(df)} записів)")
with col2:
    if st.button("Оновити всі", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

# Prepare monthly series if asked
monthly = {}
for name, df in series_dfs.items():
    if df.empty:
        monthly[name] = pd.Series(dtype=float)
        continue
    df = df.set_index('date').sort_index()
    # convert yearly to monthly by forward/backward fill or linear interpolation
    if show_interpolated:
        idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='MS')
        s = df['value'].reindex(pd.to_datetime(df.index)).sort_index()
        s_monthly = s.resample('YS').first() if False else s  # keep as-is
        # create monthly series by linear interpolation on year values
        yearly = df['value']
        yearly.index = pd.DatetimeIndex([pd.Timestamp(f'{d.year}-01-01') for d in yearly.index])
        monthly_index = pd.date_range(yearly.index.min(), yearly.index.max(), freq='MS')
        monthly_series = yearly.reindex(monthly_index, method=None)
        monthly_series = monthly_series.interpolate(method='linear')
        monthly[name] = monthly_series
    else:
        monthly[name] = df['value']

# Plot interactive charts
st.subheader("Інтерактивні графіки")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    sel = st.selectbox("Оберіть показник для графіка (ліва панель)", indicators_selected, index=0)
    s = monthly.get(sel, pd.Series())
    if s.empty:
        st.info("Немає даних для візуалізації")
    else:
        fig = px.line(x=s.index, y=s.values, labels={'x':'Дата', 'y':sel}, title=sel)
        st.plotly_chart(fig, use_container_width=True)
with chart_col2:
    sel2 = st.selectbox("Оберіть показник для графіка (права панель)", indicators_selected, index=min(1, max(0, len(indicators_selected)-1)))
    s2 = monthly.get(sel2, pd.Series())
    if s2.empty:
        st.info("Немає даних для візуалізації")
    else:
        fig2 = px.line(x=s2.index, y=s2.values, labels={'x':'Дата', 'y':sel2}, title=sel2)
        st.plotly_chart(fig2, use_container_width=True)

# Correlations
st.subheader("Кореляції між показниками")
# build DataFrame with aligned indices
aligned = pd.DataFrame()
for name, s in monthly.items():
    if s.empty:
        continue
    aligned[name] = s
aligned = aligned.dropna()
if aligned.empty or aligned.shape[1] < 2:
    st.info("Недостатньо перехресних даних для кореляції")
else:
    corr = aligned.corr()
    st.write("Матриця кореляцій (Пірсон):")
    st.dataframe(corr.style.format(precision=3))
    heatmap_fig = px.imshow(corr.values, x=corr.columns, y=corr.index, text_auto='.3f', title='Кореляційна матриця')
    st.plotly_chart(heatmap_fig, use_container_width=True)

# Forecasting: simple SARIMAX per series
st.subheader("Прогноз на 6 місяців")
forecast_col1, forecast_col2 = st.columns([2,1])
with forecast_col1:
    to_forecast = st.selectbox("Який показник прогнозувати?", indicators_selected)
    s = monthly.get(to_forecast, pd.Series())
    if s.empty or len(s.dropna()) < 12:
        st.warning("Недостатньо даних (потрібно щонайменше ~12 місячних спостережень). Пробуйте інший показник або зніміть інтерполяцію.")
    else:
        # fit SARIMAX (simple defaults) on the series
        y = s.dropna()
        try:
            model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=6)
            pred_mean = pred.predicted_mean
            pred_ci = pred.conf_int()

            # plot
            figf = px.line()
            figf.add_scatter(x=y.index, y=y.values, mode='lines', name='Історія')
            figf.add_scatter(x=pred_mean.index, y=pred_mean.values, mode='lines', name='Прогноз (6 міс.)')
            figf.update_layout(title=f'Прогноз для {to_forecast} на 6 місяців')
            st.plotly_chart(figf, use_container_width=True)

            st.write("Прогнозні значення:")
            df_pred = pd.DataFrame({"date": pred_mean.index, "forecast": pred_mean.values})
            st.dataframe(df_pred.set_index('date').round(3))

        except Exception as e:
            st.error(f"Помилка під час побудови моделі: {e}")

with forecast_col2:
    if st.button("Завантажити CSV (усі серії)"):
        # prepare combined CSV from original yearly + interpolated monthly
        out = io.StringIO()
        combined = pd.DataFrame()
        for name, s in monthly.items():
            combined[name] = s
        combined.to_csv(out)
        st.download_button("Download CSV", data=out.getvalue().encode('utf-8'), file_name='ukraine_econ_series.csv', mime='text/csv')

# Help / how to run
st.markdown("---")
st.header("Як розгорнути (швидко)")
st.markdown(
    "1. Створіть репозиторій на GitHub і додайте цей файл `ukraine_economic_dashboard.py` в корінь.\n"
    "2. Додайте `requirements.txt` з переліком залежностей (streamlit, pandas, numpy, requests, plotly, statsmodels).\n"
    "3. Запустіть локально: `pip install -r requirements.txt` -> `streamlit run ukraine_economic_dashboard.py`.\n"
    "4. Для кращих (щомісячних) даних можна підключити TradingEconomics (має API key) або державні SDMX API — я добавив посилання на джерела у чаті."
)

st.write("Готово — якщо хочете, можу згенерувати `requirements.txt` і `README.md` та підготувати інструкцію для GitHub Actions / Streamlit Community Cloud.")

# Footer
st.caption("Побудовано з даними World Bank (Indicators API). Для більш детальної / частішої (щомісячної) статистики рекомендується підключити TradingEconomics або державні SDMX API.")
