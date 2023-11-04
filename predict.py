#!/usr/bin/python3
"""
CO2 Emission Predictor API

This module provides a Flask API for predicting CO2 emissions
of a car based on the received JSON data. It uses a pre-trained
RandomForest Regressor model stored in "random_forest.bin"
for making predictions.

Usage:
    You can use this API to send JSON data containing car information
    and receive a prediction of CO2 emissions in response.

"""

import pickle
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

MODEL_FILE = "random_forest.bin"

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)


app = Flask("co2_emission")

@app.route("/predict", methods=['POST'])
def predict():
    """
    Predict CO2 emissions of a car based on the received JSON data.

    Parameters
    ----------
    car : dict
        A dictionary containing car information in JSON format.

    Returns
    -------
    dict
        A dictionary containing the predicted CO2 emission.
    """
    car = request.get_json()

    car_data = dv.transform([car])
    y_pred = model.predict(car_data)
    y_pred = np.expm1(y_pred).round(2)
    pred = float(y_pred)

    return jsonify(pred)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="9696")
