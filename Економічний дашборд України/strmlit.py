import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Заголовок ----------
st.title("📊 Економічний дашборд України")

# ---------- Синтетичні дані (імітація відкритих API) ----------
np.random.seed(42)
months = pd.date_range("2023-01-01", periods=24, freq="M")

data = pd.DataFrame({
    "Місяць": months,
    "Інфляція (%)": np.random.uniform(4, 12, size=24),
    "ВВП (млрд $)": np.random.uniform(150, 210, size=24),
    "Безробіття (%)": np.random.uniform(7, 11, size=24)
})

st.subheader("📅 Економічні показники за останні 2 роки")
st.dataframe(data)

# ---------- Інтерактивні графіки ----------
st.subheader("📈 Динаміка показників")

option = st.selectbox(
    "Виберіть показник для графіку:",
    ["Інфляція (%)", "ВВП (млрд $)", "Безробіття (%)"]
)

st.line_chart(data.set_index("Місяць")[option])

# ---------- Кореляції ----------
st.subheader("📊 Кореляція між показниками")
corr = data.drop(columns=["Місяць"]).corr()
st.dataframe(corr)

# ---------- Прогноз на 6 місяців ----------
st.subheader("🔮 Прогноз на 6 місяців")

feature = st.selectbox("Оберіть показник для прогнозу:", ["Інфляція (%)", "ВВП (млрд $)", "Безробіття (%)"])

# Побудова простої лінійної регресії
X = np.arange(len(data)).reshape(-1, 1)
y = data[feature].values
model = LinearRegression().fit(X, y)

future_X = np.arange(len(data), len(data) + 6).reshape(-1, 1)
future_pred = model.predict(future_X)

future_months = pd.date_range(data["Місяць"].iloc[-1] + pd.offsets.MonthEnd(1), periods=6, freq="M")
forecast = pd.DataFrame({"Місяць": future_months, f"Прогноз {feature}": future_pred})

st.dataframe(forecast)

st.subheader("📉 Прогнозований графік")
full_series = pd.concat([data[["Місяць", feature]], forecast.rename(columns={f"Прогноз {feature}": feature})])
st.line_chart(full_series.set_index("Місяць")[feature])

st.success("✅ Дашборд успішно згенеровано!")
