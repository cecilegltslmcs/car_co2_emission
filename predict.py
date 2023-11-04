import pickle
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

model_file = "random_forest.bin"

with open(model_file, "rb") as f_in:
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
    
    X = dv.transform([car])
    y_pred = model.predict(X)
    y_pred = np.expm1(y_pred).round(2)
    pred = float(y_pred)
    
    return jsonify(pred)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="9696")