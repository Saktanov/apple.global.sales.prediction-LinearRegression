import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Apple Sales ML", layout="wide")

# загрузка
df = pd.read_csv("apple_global_sales_dataset.csv")
model = joblib.load("model.pkl")

st.title("📊 Apple Sales Prediction App")

# ===== МЕТРИКИ =====
st.subheader("📈 Качество модели")

try:
    with open("metrics.txt", "r") as f:
        metrics = f.read()
    st.text(metrics)
except:
    st.warning("Метрики не найдены. Запусти train.py")

# ===== ГРАФИКИ =====
st.subheader("📊 Графики")

st.image("static/actual_vs_predicted.png", 
         caption="Реальные vs Предсказанные (красная линия = идеал)")



# ===== ДАННЫЕ =====
st.subheader("📄 Данные")

col1, col2 = st.columns(2)

with col1:
    st.write("С индексом")
    st.dataframe(df.head(10))

with col2:
    st.write("Без индекса")
    st.dataframe(df.head(10).reset_index(drop=True))

# ===== ПРЕДСКАЗАНИЕ =====
st.subheader("🧠 Сделать предсказание")

price = st.number_input("💰 Цена (USD)", min_value=0.0)
discount = st.number_input("🏷️ Скидка (%)", min_value=0.0)
units = st.number_input("📦 Количество продаж", min_value=0.0)

if st.button("Предсказать"):
    data = pd.DataFrame([{
        "unit_price_usd": price,
        "discount_pct": discount,
        "units_sold": units
    }])

    prediction = model.predict(data)

    st.success(f"💰 Предсказанная выручка: {prediction[0]:.2f} USD")