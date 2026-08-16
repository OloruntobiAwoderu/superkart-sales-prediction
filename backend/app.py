
import joblib
import pandas as pd
from flask import Flask, jsonify, request

superkart_api = Flask("SuperKart")

# The model file is copied into /app by the backend Dockerfile.
model = joblib.load("superkart_model.joblib")

FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


@superkart_api.get("/")
def home():
    return jsonify(
        {
            "message": "Welcome to the SuperKart System",
            "endpoints": ["/v1/predict", "/v1/predictbatch"],
        }
    )


@superkart_api.get("/health")
def health():
    return jsonify({"status": "ok"})


@superkart_api.post("/v1/predict")
def predict_sales():
    data = request.get_json(silent=True) or {}
    missing = [column for column in FEATURE_COLUMNS if column not in data]

    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        input_data = pd.DataFrame(
            [{column: data[column] for column in FEATURE_COLUMNS}]
        )
        prediction = float(model.predict(input_data)[0])
        return jsonify({"Sales": round(prediction, 2)})
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 400


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file using the 'file' form field."}), 400

    try:
        input_data = pd.read_csv(request.files["file"])
        missing = [column for column in FEATURE_COLUMNS if column not in input_data.columns]
        if missing:
            return jsonify({"error": f"CSV is missing required columns: {missing}"}), 400

        predictions = model.predict(input_data[FEATURE_COLUMNS])
        output = [
            {"row": int(index), "Sales": round(float(prediction), 2)}
            for index, prediction in enumerate(predictions)
        ]
        return jsonify(output)
    except Exception as exc:
        return jsonify({"error": f"Batch prediction failed: {exc}"}), 400


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
