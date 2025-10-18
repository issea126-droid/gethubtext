import streamlit as st 
import pandas as pd
import requests
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Економічний дашборд України", layout="wide")

WB_BASE = "https://api.worldbank.org/v2/country/UA/indicator/{}"

@st.cache_data(ttl=60*60)
def fetch_worldbank_indicator(indicator_code, per_page=1000):
    url = WB_BASE.format(indicator_code)
    params = {"format": "json", "per_page": per_page}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    rows = payload[1] if len(payload) > 1 else []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    df['indicator'] = df['indicator'].apply(lambda x: x.get('id') if isinstance(x, dict) else x)
    df = df[['date', 'value']]
    df['date'] = df['date'].astype(int)
    df.rename(columns={'value': indicator_code}, inplace=True)
    df[indicator_code] = pd.to_numeric(df[indicator_code], errors='coerce')
    return df.sort_values('date')

# --- Sidebar ---
st.sidebar.title("Налаштування джерел даних")
use_worldbank = st.sidebar.checkbox("Використовувати World Bank API", value=True)

if not use_worldbank:
    st.warning("Ви відключили World Bank API. Дашборд поки що підтримує World Bank.")

# --- Load data ---
with st.spinner("Завантаження даних..."):
    if use_worldbank:
        try:
            gdp = fetch_worldbank_indicator("NY.GDP.MKTP.CD")
            inflation = fetch_worldbank_indicator("FP.CPI.TOTL.ZG")
            unemployment = fetch_worldbank_indicator("SL.UEM.TOTL.ZS")

            dfs = [gdp, inflation, unemployment]
            df_merged = None
            for d in dfs:
                if df_merged is None:
                    df_merged = d
                else:
                    df_merged = pd.merge(df_merged, d, on=['date'], how='outer')

            df_merged.rename(columns={'date': 'year'}, inplace=True)
            df_merged['year'] = df_merged['year'].astype(int)
            df_merged.sort_values('year', inplace=True)
            df = df_merged
        except Exception as e:
            st.error(f"Помилка при завантаженні даних: {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

# --- Title ---
st.title("📊 Економічний дашборд України")

# --- Layout ---
col1, col2 = st.columns([1,3])

with col1:
    st.subheader("Фільтри")
    years = df['year'].dropna().astype(int).tolist() if not df.empty else []
    if years:
        min_year = min(years)
        max_year = max(years)
        selected_range = st.slider("Рік (діапазон):", min_value=min_year, max_value=max_year, value=(max(min_year, max_year-10), max_year))
    else:
        selected_range = (2000, datetime.now().year)

    selected_metrics = st.multiselect(
        "Метрики для графіка:", 
        ["GDP (US$)", "Inflation (annual %)", "Unemployment (%)"], 
        default=["GDP (US$)"]
    )
    show_table = st.checkbox("Показати таблицю даних", value=False)

with col2:
    st.subheader("Огляд")
    if df.empty:
        st.info("Немає завантажених даних.")
    else:
        latest = df.dropna(subset=['NY.GDP.MKTP.CD','FP.CPI.TOTL.ZG','SL.UEM.TOTL.ZS']).iloc[-1: ]
        k1, k2, k3 = st.columns(3)
        k1.metric("GDP (current US$)", f"{int(latest['NY.GDP.MKTP.CD'].values[0]):,}" if pd.notna(latest['NY.GDP.MKTP.CD'].values[0]) else "—")
        k2.metric("Inflation (annual %)", f"{latest['FP.CPI.TOTL.ZG'].values[0]:.2f}%" if pd.notna(latest['FP.CPI.TOTL.ZG'].values[0]) else "—")
        k3.metric("Unemployment (%)", f"{latest['SL.UEM.TOTL.ZS'].values[0]:.2f}%" if pd.notna(latest['SL.UEM.TOTL.ZS'].values[0]) else "—")

        yr_mask = (df['year'] >= selected_range[0]) & (df['year'] <= selected_range[1])
        df_viz = df.loc[yr_mask].copy()

        # --- Побудова інтерактивних графіків ---
        if not df_viz.empty and selected_metrics:
            df_long = pd.DataFrame()
            for metric in selected_metrics:
                if metric == "GDP (US$)":
                    y_col = "NY.GDP.MKTP.CD"
                elif metric == "Inflation (annual %)":
                    y_col = "FP.CPI.TOTL.ZG"
                else:
                    y_col = "SL.UEM.TOTL.ZS"
                temp = df_viz[['year', y_col]].copy()
                temp = temp.rename(columns={y_col: 'value'})
                temp['metric'] = metric
                df_long = pd.concat([df_long, temp])

            df_long = df_long.dropna(subset=['value'])

            chart = alt.Chart(df_long).mark_line(point=True).encode(
                x=alt.X('year:O', title='Year'),
                y=alt.Y('value:Q', title='Value'),
                color='metric:N',
                tooltip=['year', 'metric', alt.Tooltip('value:Q', format=',.2f')]
            ).interactive().properties(height=400)

            st.altair_chart(chart, use_container_width=True)

        if show_table:
            st.dataframe(df_viz)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Завантажити CSV", data=csv, file_name='ukraine_economic_data.csv', mime='text/csv')

st.markdown("---")
st.caption("Дані завантажуються через World Bank API: https://api.worldbank.org")
