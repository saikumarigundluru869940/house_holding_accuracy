import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("house_price_model.pkl")
info = json.load(open("model_info.json"))

NUM_FIELDS = ["area_sqft", "bedrooms", "bathrooms", "age_years",
              "floor_number", "parking_spots", "distance_to_center_km", "has_lift"]
COLUMNS = ["city", "area_sqft", "bedrooms", "bathrooms", "age_years", "floor_number",
           "parking_spots", "distance_to_center_km", "furnishing_status", "has_lift"]


@app.get("/")
def home():
    return jsonify(
        status="Backend is running",
        try_these=["/api/info", "POST /api/predict"],
    )


@app.errorhandler(404)
def not_found(e):
    return jsonify(error="This address does not exist. Use /api/info or POST /api/predict."), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify(error="Something went wrong on the server. Check the Flask terminal for details."), 500


@app.get("/api/info")
def get_info():
    return jsonify(info)


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        row = {f: float(data[f]) for f in NUM_FIELDS}
        row["city"] = data["city"]
        row["furnishing_status"] = data["furnishing_status"]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="Fill in every field with valid values."), 400
    if row["city"] not in info["cities"] or row["furnishing_status"] not in info["furnishing"]:
        return jsonify(error="Unknown city or furnishing status."), 400

    price = float(model.predict(pd.DataFrame([row])[COLUMNS])[0])
    result = {"predicted_price_lakhs": round(price, 2)}

    actual = data.get("actual_price")
    if actual not in (None, ""):
        try:
            actual = float(actual)
            result["actual_price_lakhs"] = actual
            result["error_lakhs"] = round(price - actual, 2)
            result["error_percent"] = round(abs(price - actual) / actual * 100, 2)
        except (ValueError, ZeroDivisionError):
            return jsonify(error="Actual price must be a positive number."), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=8000, debug=False)