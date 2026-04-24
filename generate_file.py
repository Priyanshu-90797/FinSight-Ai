import pandas as pd
import random
from datetime import datetime, timedelta

# ---------------- SETTINGS ----------------
start_date = datetime(2026, 1, 1)
days = 365  # 1 year data

# ---------------- CATEGORIES ----------------
categories = {
    "Food": (200, 1000),
    "Travel": (300, 2000),
    "Shopping": (500, 5000),
    "Bills": (1000, 5000),
    "Entertainment": (300, 2000),
    "Rent": (8000, 20000),
    "Health": (200, 5000),
    "Education": (500, 10000),
    "Subscriptions": (100, 1000),
    "Insurance": (500, 5000),
    "EMI": (2000, 15000),
    "Investments": (1000, 20000)
}

# ---------------- INCOME TYPES ----------------
income_types = {
    "salary": (20000, 100000),
    "freelance": (2000, 20000),
    "business": (5000, 50000),
    "passive_income": (1000, 15000)
}

data = []

# ---------------- GENERATE DATA ----------------
for i in range(days):
    date = start_date + timedelta(days=i)

    # Income generation
    income = 0
    income_source = "none"

    # Monthly salary
    if date.day == 1:
        income = random.randint(*income_types["salary"])
        income_source = "salary"

    # Random side income
    elif random.random() < 0.2:
        source = random.choice(["freelance", "business", "passive_income"])
        income = random.randint(*income_types[source])
        income_source = source

    # Expense generation
    category = random.choice(list(categories.keys()))
    amount = random.randint(*categories[category])

    # ---------------- SAVINGS LOGIC ----------------
    savings_flag = 0

    # High spending categories
    if category in ["Shopping", "Entertainment", "Food"] and amount > 1200:
        savings_flag = 1

    # Too many subscriptions
    if category == "Subscriptions" and amount > 500:
        savings_flag = 1

    # Overspending on travel
    if category == "Travel" and amount > 1500:
        savings_flag = 1

    # ---------------- APPEND DATA ----------------
    data.append({
        "date": date.strftime("%Y-%m-%d"),
        "income": income,
        "income_source": income_source,
        "category": category,
        "expense": amount,
        "is_saving_possible": savings_flag
    })

# ---------------- CREATE DATAFRAME ----------------
df = pd.DataFrame(data)

# ---------------- ADD FEATURES ----------------
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["year"] = df["date"].dt.year
df["weekday"] = df["date"].dt.day_name()

# ---------------- SAVINGS ESTIMATION ----------------
df["net_savings"] = df["income"] - df["expense"]

# ---------------- SAVE ----------------
df.to_csv("data/finance_dataset.csv", index=False)

print("✅ Advanced dataset generated: data/finance_dataset.csv")

