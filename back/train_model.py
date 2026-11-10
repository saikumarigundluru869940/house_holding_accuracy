import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("house_price_eda_dataset.csv")
target = "price_lakhs"
cat_cols = ["city", "furnishing_status"]

X = df.drop(columns=[target, "house_id"])  # house_id is just an ID, not a feature
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def build(model):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", model)])


models = {
    "Linear Regression": build(LinearRegression()),
    "Random Forest": build(RandomForestRegressor(n_estimators=300, random_state=42)),
}

metrics = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    metrics[name] = {
        "r2": round(r2_score(y_test, pred), 4),
        "mae": round(mean_absolute_error(y_test, pred), 2),
    }
    print(name, metrics[name])

best = max(metrics, key=lambda k: metrics[k]["r2"])  # keep the better model
joblib.dump(models[best], "house_price_model.pkl")
info = {
    "metrics": metrics,
    "model_used": best,
    "cities": sorted(df["city"].unique().tolist()),
    "furnishing": sorted(df["furnishing_status"].unique().tolist()),
}
json.dump(info, open("model_info.json", "w"), indent=2)
print("Saved house_price_model.pkl and model_info.json")