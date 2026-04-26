import pandas as pd

class FinSightAdvisor:

    def __init__(self, finance_context, df):
        self.context = finance_context
        self.df = df

    def chat(self, question):

        question = question.lower()

        
        if self.df.empty:
            return "No data available. Please upload or generate data first."

        total_income = self.df['income'].sum()
        total_expense = self.df['expense'].sum()

        if total_expense == 0:
            return "No expense data found."

        cat = self.df.groupby('category')['expense'].sum()

        top_cat = cat.idxmax()
        top_val = cat.max()

        savings = total_income - total_expense
        savings_rate = (savings / total_income) * 100 if total_income > 0 else 0

    
        if "save" in question or "saving" in question:

            suggestions = []
            total_possible_savings = 0

            for c, v in cat.items():
                percent = (v / total_expense) * 100

                if percent > 20:
                    reduce_amt = int(v * 0.2)
                    total_possible_savings += reduce_amt

                    suggestions.append(
                        f"• {c}: ₹{int(v)} ({round(percent,1)}%) → Cut 20% → Save ₹{reduce_amt}"
                    )

            return f"""
💡 **Personalized Savings Plan**

💰 Total Spending: ₹{int(total_expense)}  
📊 Top Category: {top_cat} (₹{int(top_val)})

📉 Suggested Reductions:
{chr(10).join(suggestions) if suggestions else "• Your spending is already well balanced"}

🎯 Potential Monthly Savings: ₹{total_possible_savings}

🔥 Tip: Focus on reducing {top_cat} first for maximum impact
"""

    
        elif "overspend" in question or "spending" in question:

            analysis = []

            for c, v in cat.items():
                percent = (v / total_expense) * 100

                if percent > 35:
                    status = "🚨 Critical"
                elif percent > 25:
                    status = "⚠️ High"
                else:
                    status = "✅ Normal"

                analysis.append(f"{c}: {round(percent,1)}% → {status}")

            return f"""
📊 **Spending Analysis Report**

💰 Total Expense: ₹{int(total_expense)}

📌 Category Breakdown:
{chr(10).join(analysis)}

🔥 Highest spending: {top_cat} (₹{int(top_val)})
👉 Recommendation: Reduce this category first
"""

    
        elif "future" in question or "predict" in question:

            monthly = self.df.groupby(self.df['date'].dt.to_period('M'))['expense'].sum()

            if len(monthly) >= 2:
                last = monthly.iloc[-1]
                prev = monthly.iloc[-2]

                growth = (last - prev) / prev if prev != 0 else 0
                prediction = int(last * (1 + growth))

                trend = "increasing 📈" if growth > 0 else "decreasing 📉"

                return f"""
📈 **Future Expense Prediction**

Last Month: ₹{int(last)}  
Previous Month: ₹{int(prev)}

Growth Trend: {round(growth*100,1)}% ({trend})

👉 Expected Next Month: ₹{prediction}

⚠️ If trend continues, your spending will {'increase' if growth > 0 else 'decrease'}
"""

            return "⚠️ Not enough data to predict future expenses."

        
        elif "health" in question:

            if savings_rate >= 20:
                status = "🟢 Excellent"
            elif savings_rate >= 10:
                status = "🟡 Moderate"
            else:
                status = "🔴 Poor"

            return f"""
🧠 **Financial Health Report**

💰 Income: ₹{int(total_income)}  
💸 Expenses: ₹{int(total_expense)}  
💾 Savings: ₹{int(savings)}

📊 Savings Rate: {round(savings_rate,1)}%  
Status: {status}

🎯 Recommended: Save at least 20% of income
"""

        
        else:
            return f"""
🤖 **FinSight AI Assistant**

I can help you with:

• 💰 Savings optimization  
• 📊 Spending analysis  
• 📈 Future predictions  
• 🧠 Financial health  

👉 Try asking:
- "How can I save money?"
- "Where am I overspending?"
- "Predict my future expenses"
- "Check my financial health"
"""
