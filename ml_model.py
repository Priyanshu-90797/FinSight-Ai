import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

MODEL_PATH = Path("models/model.pkl")


# 🔥 FEATURE ENGINEERING
def prepare_features(df):
    df = df.copy()

    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday

    return df


# 🚀 TRAIN MODEL
def train_model(df):

    df = prepare_features(df)

    X = df[["month", "day", "weekday"]]
    y = df["expense"]

    # 🔥 train-test split (important)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3
    )

    model.fit(X_train, y_train)

    # 🔥 predictions
    y_pred = model.predict(X_test)

    # 🔥 metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 🔥 save model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {
        "pipeline": model,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_mae": mae,
            "feature_importance": dict(
                zip(X.columns, model.feature_importances_)
            )
        },
        "y_test": y_test,
        "y_pred": y_pred
    }


# 🚀 LOAD MODEL (SAFE)
def load_pipeline():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError("Model not trained yet. Run training first.")


# 🔮 PREDICT MONTH EXPENSES
def predict_month_expenses(year, month, model):

    # 🔥 correct days in month
    dates = pd.date_range(
        start=f"{year}-{month:02d}-01",
        end=f"{year}-{month:02d}-28"
    )

    df = pd.DataFrame({"date": dates})
    df = prepare_features(df)

    df["predicted_expense"] = model.predict(
        df[["month", "day", "weekday"]]
    )

    # 🔥 smoothing (realistic prediction)
    df["predicted_expense"] = (
    df["predicted_expense"]
    .rolling(window=3, min_periods=1)
    .mean()
)

    return df  