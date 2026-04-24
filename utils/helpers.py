import pandas as pd

def load_data():
    df = pd.read_csv("data/finance_dataset.csv")

    # date convert
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # required columns only
    required_cols = ["date", "income", "income_source", "category", "expense"]
    df = df[[col for col in required_cols if col in df.columns]]

    # numeric conversion
    df["income"] = pd.to_numeric(df["income"], errors="coerce").fillna(0)
    df["expense"] = pd.to_numeric(df["expense"], errors="coerce").fillna(0)

    # drop invalid dates
    df = df.dropna(subset=["date"])

    # derived features
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["weekday"] = df["date"].dt.day_name()

    return df


# 📊 KPI CALCULATION
def compute_kpis(df):
    total_income = df["income"].sum()
    total_expense = df["expense"].sum()
    net_savings = total_income - total_expense

    return {
        "total_income": total_income,
        "total_expense": total_expense,
        "net_savings": net_savings,
        "avg_daily_expense": df["expense"].mean(),
        "savings_rate": net_savings / total_income if total_income else 0
    }


# 📈 MONTHLY SUMMARY (FIXED 💀)
def get_monthly_summary(df):
    df = df.copy()

    df["month_year"] = df["date"].dt.to_period("M").astype(str)

    grouped = df.groupby("month_year", as_index=False).agg({
        "income": "sum",
        "expense": "sum"
    })

    # 🔥 IMPORTANT FIX
    grouped.rename(columns={
        "income": "total_income",
        "expense": "total_expense"
    }, inplace=True)

    grouped["net_savings"] = grouped["total_income"] - grouped["total_expense"]

    return grouped


# 📊 CATEGORY ANALYSIS
def get_category_summary(df):
    cat = df.groupby("category")["expense"].sum().reset_index()
    total = cat["expense"].sum()

    cat["total"] = cat["expense"]
    cat["share_pct"] = (cat["expense"] / total * 100).round(1)

    return cat.sort_values("expense", ascending=False)


# 💰 INCOME SOURCES
def get_income_source_summary(df):
    if "income_source" not in df.columns:
        return pd.DataFrame()

    return df.groupby("income_source")["income"].sum().reset_index()


# 🚨 SMART ALERTS
def detect_overspending(df):
    cat = get_category_summary(df)
    alerts = []

    for _, row in cat.iterrows():
        if row["share_pct"] > 20:
            alerts.append({
                "category": row["category"],
                "total": row["total"],
                "avg_daily": row["total"] / len(df),
                "share_pct": row["share_pct"],
                "severity": "high" if row["share_pct"] > 30 else "medium"
            })

    return alerts


# 💡 SAVINGS ANALYSIS
def savings_opportunity_report(df):
    kpis = compute_kpis(df)

    monthly = df.groupby(df["date"].dt.to_period("M")).agg({
        "income": "sum",
        "expense": "sum"
    })

    monthly["savings"] = monthly["income"] - monthly["expense"]

    return {
        "actual_rate": kpis["savings_rate"],
        "target_rate": 0.2,
        "on_track": kpis["savings_rate"] >= 0.2,
        "total_months": len(monthly),
        "negative_months": (monthly["savings"] < 0).sum(),
        "best_month": str(monthly["savings"].idxmax()),
        "worst_month": str(monthly["savings"].idxmin())
    }


# 💰 FORMATTERS
# WHAT-IF SIMULATION
def multi_simulation(df, changes):
    """
    Run a multi-category what-if simulation.

    Example:
        multi_simulation(df, {"Food": -20, "Shopping": -10})

    ``changes`` values are percentages. Negative values reduce spending and
    positive values increase spending. Category names are matched
    case-insensitively, but the original category names are preserved.
    """
    if df is None or df.empty:
        return {"error": "No data"}

    required_cols = {"income", "expense", "category"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return {"error": f"Missing required columns: {', '.join(sorted(missing_cols))}"}

    if not isinstance(changes, dict) or not changes:
        return {"error": "No category changes provided"}

    work = df.copy()
    work["income"] = pd.to_numeric(work["income"], errors="coerce").fillna(0)
    work["expense"] = pd.to_numeric(work["expense"], errors="coerce").fillna(0)
    work["category"] = work["category"].fillna("Uncategorized").astype(str)

    total_income = float(work["income"].sum())
    total_expense = float(work["expense"].sum())
    old_savings = total_income - total_expense
    old_savings_rate = old_savings / total_income if total_income else 0

    cat_spend = work.groupby("category")["expense"].sum().sort_values(ascending=False)
    category_lookup = {str(cat).strip().lower(): cat for cat in cat_spend.index}

    new_total = total_expense
    category_impacts = []
    ignored_categories = []

    for raw_cat, raw_pct in changes.items():
        category_key = str(raw_cat).strip().lower()
        matched_cat = category_lookup.get(category_key)

        try:
            pct = float(raw_pct)
        except (TypeError, ValueError):
            ignored_categories.append({
                "category": raw_cat,
                "reason": "Invalid percentage"
            })
            continue

        if matched_cat is None:
            ignored_categories.append({
                "category": raw_cat,
                "reason": "Category not found"
            })
            continue

        old_val = float(cat_spend[matched_cat])
        new_val = max(0, old_val * (1 + pct / 100))
        delta = new_val - old_val
        new_total += delta

        category_impacts.append({
            "category": matched_cat,
            "change_pct": round(pct, 2),
            "old_expense": int(round(old_val)),
            "new_expense": int(round(new_val)),
            "expense_change": int(round(delta)),
            "savings_impact": int(round(-delta)),
            "share_before_pct": round((old_val / total_expense) * 100, 1) if total_expense else 0,
            "share_after_pct": 0
        })

    new_savings = total_income - new_total
    new_savings_rate = new_savings / total_income if total_income else 0

    for item in category_impacts:
        item["share_after_pct"] = (
            round((item["new_expense"] / new_total) * 100, 1)
            if new_total else 0
        )

    best_impact = max(
        category_impacts,
        key=lambda item: item["savings_impact"],
        default=None
    )

    return {
        "old_expense": int(round(total_expense)),
        "new_expense": int(round(new_total)),
        "expense_change": int(round(new_total - total_expense)),
        "old_savings": int(round(old_savings)),
        "new_savings": int(round(new_savings)),
        "savings_change": int(round(new_savings - old_savings)),
        "old_savings_rate": round(old_savings_rate * 100, 1),
        "new_savings_rate": round(new_savings_rate * 100, 1),
        "savings_rate_change": round((new_savings_rate - old_savings_rate) * 100, 1),
        "category_impacts": category_impacts,
        "ignored_categories": ignored_categories,
        "best_move": best_impact["category"] if best_impact else None,
        "message": (
            f"This plan improves savings by {fmt_inr(new_savings - old_savings)}"
            if new_savings >= old_savings
            else f"This plan reduces savings by {fmt_inr(old_savings - new_savings)}"
        )
    }


# FORMATTERS
def fmt_inr(x, compact=False):
    return f"₹{x:,.0f}"

def fmt_pct(x):
    return f"{x*100:.1f}%"


# 🤖 AI CONTEXT
def build_finance_context(df):
    total_income = df["income"].sum()
    total_expense = df["expense"].sum()

    cat = df.groupby("category")["expense"].sum()

    if len(cat) == 0:
        return "No financial data available."

    top_cat = cat.idxmax()
    top_val = cat.max()

    savings = total_income - total_expense
    savings_rate = (savings / total_income) * 100 if total_income > 0 else 0

    return f"""
Financial Summary:

Total Income: ₹{int(total_income)}
Total Expense: ₹{int(total_expense)}
Savings: ₹{int(savings)}
Savings Rate: {round(savings_rate,1)}%

Top Spending Category: {top_cat} (₹{int(top_val)})

Category Breakdown:
{cat.to_dict()}
"""


def advanced_alerts(df):
    alerts = []

    # ---- 1. RUN RATE (month overspend prediction)
    daily_avg = df["expense"].mean()
    days_passed = df["date"].dt.day.max()
    projected = daily_avg * 30

    if projected > df["income"].sum() * 0.8:
        alerts.append(f"⚠️ At current pace, you may spend ₹{int(projected)} this month")

    # ---- 2. RECENT SPIKE (last 7 vs previous 7)
    df_sorted = df.sort_values("date")
    last7 = df_sorted.tail(7)["expense"].sum()
    prev7 = df_sorted.iloc[-14:-7]["expense"].sum()

    if prev7 > 0:
        change = (last7 - prev7) / prev7
        if change > 0.25:
            alerts.append(f"📈 Spending increased {round(change*100,1)}% in last 7 days")

    # ---- 3. STREAK (high spend days)
    df_sorted["high"] = df_sorted["expense"] > df_sorted["expense"].mean()
    streak = 0
    max_streak = 0

    for val in df_sorted["high"]:
        if val:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    if max_streak >= 3:
        alerts.append(f"🔥 You had {max_streak} consecutive high-spending days")

    return alerts


def generate_insights(df):
    insights = []

    total_income = df["income"].sum()
    total_expense = df["expense"].sum()

    if total_income == 0:
        return ["No income data available"]

    # 1. Top category spending
    cat = df.groupby("category")["expense"].sum()

    if len(cat) > 0:
        top_cat = cat.idxmax()
        top_val = cat.max()
        pct = (top_val / total_expense) * 100

        if pct > 30:
            insights.append(
                f"⚠️ You are spending {round(pct,1)}% on {top_cat}. Reduce by ₹{int(top_val*0.2)}"
            )

    # 2. Low savings
    savings = total_income - total_expense
    savings_rate = savings / total_income

    if savings_rate < 0.2:
        insights.append(
            f"💡 Savings rate is low ({round(savings_rate*100,1)}%). Target 20%"
        )

    # 3. Weekly trend
    df_sorted = df.sort_values("date")

    if len(df_sorted) >= 14:
        last7 = df_sorted.tail(7)["expense"].sum()
        prev7 = df_sorted.iloc[-14:-7]["expense"].sum()

        if prev7 > 0:
            change = (last7 - prev7) / prev7
            if change > 0.2:
                insights.append(
                    f"📈 Spending increased {round(change*100,1)}% this week"
                )

    return insights
