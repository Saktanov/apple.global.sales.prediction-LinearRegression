import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import os

# загрузка
df = pd.read_csv("apple_global_sales_dataset.csv")

df = df[[
    "unit_price_usd",
    "discount_pct",
    "units_sold",
    "revenue_usd"
]].dropna()

X = df[["unit_price_usd", "discount_pct", "units_sold"]]
y = df["revenue_usd"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# модель
model = LinearRegression()
model.fit(X_train, y_train)

# предсказание
y_pred = model.predict(X_test)

# 📊 график: реальные vs предсказанные
plt.figure()

plt.scatter(y_test, y_pred)

# 🔴 диагональная линия (идеал)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], color="red")

plt.xlabel("Реальные значения")
plt.ylabel("Предсказанные значения")
plt.title("Реальные vs Предсказанные")

plt.savefig("static/actual_vs_predicted.png")

# 📊 метрики
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.2f}")

# сохраняем метрики
with open("metrics.txt", "w") as f:
    f.write(f"R2: {r2:.4f}\n")
    f.write(f"MAE: {mae:.2f}")

# сохраняем модель
joblib.dump(model, "model.pkl")

# папка для графиков
os.makedirs("static", exist_ok=True)

# график 1
plt.figure()
plt.scatter(df["unit_price_usd"], df["revenue_usd"])
plt.xlabel("Цена")
plt.ylabel("Выручка")
plt.title("Цена vs Выручка")
plt.savefig("static/price_vs_revenue.png")

# график 2
plt.figure()
plt.scatter(df["units_sold"], df["revenue_usd"])
plt.xlabel("Количество")
plt.ylabel("Выручка")
plt.title("Количество vs Выручка")
plt.savefig("static/units_vs_revenue.png")

print("✅ Всё готово!")