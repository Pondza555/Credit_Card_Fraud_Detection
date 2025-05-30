import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("xgb_deploy.pkl", "rb"))
scaler = pickle.load(open("rob_scal.pkl", "rb"))
scale_cols = ["Hour","Amount"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api",methods=["POST"])
def predict_api():
    data = request.json["data"]
    df = pd.DataFrame([data])
    df["Hour"] = scaler["Hour"].transform(df[["Hour"]])
    df["Amount"] = scaler["Amount"].transform(df[["Amount"]])
    proba = model.predict_proba(df)[0][1]
    prediction = int(proba >= 0.2)
    print("proba =", proba)
    return jsonify({"prediction": prediction})

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    feature_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                     'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Hour', 'Amount']
    
    df = pd.DataFrame([data], columns=feature_names)
    df["Hour"] = scaler["Hour"].transform(df[["Hour"]])
    df["Amount"] = scaler["Amount"].transform(df[["Amount"]])

    proba = model.predict_proba(df)[0][1]
    prediction = int(proba >= 0.2)
    result_text = "Fraud" if prediction == 1 else "Non-Fraud"
    threshold_info = f"({proba:.3f} {'>' if proba >= 0.2 else '<'} 0.2 → {result_text})"

    return render_template(
        "home.html", 
        predict_text=f"Fraud Prediction: {result_text} {threshold_info} (prob={proba:.3f})"
        )

if __name__ == "__main__":
    app.run(debug=True)