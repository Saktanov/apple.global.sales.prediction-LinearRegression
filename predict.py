import joblib
import pandas as pd

model = joblib.load("model.pkl")

def get_float(prompt):
    while True:
        value = input(prompt)
        try:
            return float(value)
        except:
            print("❌ Ошибка: введите число!")

print("\n📥 Введите данные для предсказания:\n")

price = get_float("💰 Цена (USD): ")
discount = get_float("🏷️ Скидка (%): ")
units = get_float("📦 Количество продаж: ")

# ✅ создаём DataFrame с теми же именами
data = pd.DataFrame([{
    "unit_price_usd": price,
    "discount_pct": discount,
    "units_sold": units
}])

prediction = model.predict(data)

print("\n✅ Предсказанная выручка:")
print(f"{prediction[0]:.2f} USD")