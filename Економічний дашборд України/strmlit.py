import streamlit as st
import pandas as pd
import requests
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Економічний дашборд України", layout="wide")

# ----------------------
# Helper: World Bank API
# ----------------------
WB_BASE = "https://api.worldbank.org/v2/country/UA/indicator/{}"

@st.cache_data(ttl=60*60)
def fetch_worldbank_indicator(indicator_code, per_page=1000):
    """Fetch an indicator for Ukraine from the World Bank API and return a tidy DataFrame.
    indicator_code examples:
      - NY.GDP.MKTP.CD  -> GDP (current US$)
      - FP.CPI.TOTL.ZG  -> Inflation, consumer prices (annual %)
      - SL.UEM.TOTL.ZS  -> Unemployment, total (% of total labor force)
    """
    url = WB_BASE.format(indicator_code)
    params = {"format": "json", "per_page": per_page}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    # payload[1] contains data rows
    rows = payload[1] if len(payload) > 1 else []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    # Clean and select columns we want
    df = df[["date", "value", "indicator"]]
    df["date"] = df["date"].astype(int)
    df.rename(columns={"value": indicator_code}, inplace=True)
    df[indicator_code] = pd.to_numeric(df[indicator_code], errors="coerce")
    return df.sort_values("date")

# ----------------------
# Load data
# ----------------------
st.sidebar.title("Налаштування джерел даних")
use_worldbank = st.sidebar.checkbox("Використовувати World Bank API (за замовчуванням)", value=True)

# optional: placeholders for alternative APIs
st.sidebar.markdown("---")
st.sidebar.write("Інші джерела (опційно): NBU / StateStatistics / TradingEconomics — потребують API-ключі або інший формат.")

if not use_worldbank:
    st.warning("Ви відключили World Bank API. Дашборд поки що підтримує World Bank як основне джерело. Увімкніть його або розширте код для інших API.")

with st.spinner("Завантаження даних..."):
    if use_worldbank:
        try:
            gdp = fetch_worldbank_indicator("NY.GDP.MKTP.CD")
            inflation = fetch_worldbank_indicator("FP.CPI.TOTL.ZG")
            unemployment = fetch_worldbank_indicator("SL.UEM.TOTL.ZS")

            # merge into one dataframe by date
            dfs = [gdp, inflation, unemployment]
            df_merged = None
            for d in dfs:
                if df_merged is None:
                    df_merged = d
                else:
                    df_merged = pd.merge(df_merged, d, on=["date", "indicator"], how="outer")

            # The merge approach above keeps indicator column duplicated; instead rebuild tidy table
            df = pd.DataFrame({"year": sorted(set(gdp["date"].tolist() + inflation["date"].tolist() + unemployment["date"].tolist()))})
            df = df.set_index("year")

            def series_from(ind_df, code):
                if ind_df.empty:
                    return pd.Series(dtype=float)
                s = ind_df.set_index("date")[code]
                s.index = s.index.astype(int)
                return s

            df = df.join(series_from(gdp, "NY.GDP.MKTP.CD"), how="left")
            df = df.join(series_from(inflation, "FP.CPI.TOTL.ZG"), how="left")
            df = df.join(series_from(unemployment, "SL.UEM.TOTL.ZS"), how="left")
            df.reset_index(inplace=True)
            df.rename(columns={"index": "year"}, inplace=True)
            df["year"] = df["year"].astype(int)
            df.sort_values("year", inplace=True)
        except Exception as e:
            st.error(f"Помилка при завантаженні даних: {e}")
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

# ----------------------
# UI layout
# ----------------------
st.title("📊 Економічний дашборд України")
st.markdown("Дані завантажуються з відкритих API (World Bank). Ви можете розширити джерела у коді.")

col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("Фільтри")
    years = df["year"].dropna().astype(int).tolist() if not df.empty else []
    if years:
        min_year = int(min(years))
        max_year = int(max(years))
        selected_range = st.slider("Рік (діапазон):", min_value=min_year, max_value=max_year, value=(max_year-10 if max_year-10>min_year else min_year, max_year))
    else:
        selected_range = (2000, datetime.now().year)

    metric = st.radio("Метрика:", ("GDP (US$)", "Inflation (annual %)", "Unemployment (%)"))
    show_table = st.checkbox("Показати таблицю даних", value=False)
    st.markdown("---")
    st.markdown("**Поради:** Ви можете завантажити CSV нижче або додати додаткові показники (NBU, Stat.gov.ua).")

with col2:
    st.subheader("Огляд")
    if df.empty:
        st.info("Немає завантажених даних. Перевірте налаштування джерел або підключення до інтернету.")
    else:
        # KPI cards
        latest = df.dropna(subset=["NY.GDP.MKTP.CD", "FP.CPI.TOTL.ZG", "SL.UEM.TOTL.ZS"]).iloc[-1:]
        k1, k2, k3 = st.columns(3)
        try:
            k1.metric("GDP (current US$)", f"{int(latest['NY.GDP.MKTP.CD'].values[0]):,}")
        except Exception:
            k1.metric("GDP (current US$)", "—")
        try:
            k2.metric("Inflation (annual %)", f"{latest['FP.CPI.TOTL.ZG'].values[0]:.2f}%")
        except Exception:
            k2.metric("Inflation (annual %)", "—")
        try:
            k3.metric("Unemployment (%)", f"{latest['SL.UEM.TOTL.ZS'].values[0]:.2f}%")
        except Exception:
            k3.metric("Unemployment (%)", "—")

        # Filter by selected range
        yr_mask = (df["year"] >= int(selected_range[0])) & (df["year"] <= int(selected_range[1]))
        df_viz = df.loc[yr_mask].copy()

        # Choose metric to plot
        if metric == "GDP (US$)":
            y = "NY.GDP.MKTP.CD"
            y_title = "GDP (current US$)"
        elif metric == "Inflation (annual %)":
            y = "FP.CPI.TOTL.ZG"
            y_title = "Inflation, annual %"
        else:
            y = "SL.UEM.TOTL.ZS"
            y_title = "Unemployment %"

        chart = alt.Chart(df_viz).mark_line(point=True).encode(
            x=alt.X('year:O', title='Year'),
            y=alt.Y(f'{y}:Q', title=y_title),
            tooltip=['year', alt.Tooltip(f'{y}:Q', format=',.2f')]
        ).properties(height=400)

        st.altair_chart(chart, use_container_width=True)

        if show_table:
            st.dataframe(df_viz)

        # CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Завантажити CSV з усіма даними", data=csv, file_name='ukraine_economic_data.csv', mime='text/csv')

# ----------------------
# Footer / Notes
# ----------------------
st.markdown("---")
st.caption("Дані (за замовчуванням) завантажуються через World Bank API: https://api.worldbank.org. Ви можете додати інші джерела як NBU або State Statistics (stat.gov.ua). Для реального часу або частіших оновлень розгляньте API TradingEconomics або власні ETL-процеси.")

