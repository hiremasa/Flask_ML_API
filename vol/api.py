
from joblib import load
import flask
import numpy as np
import pandas as pd
import lightgbm as lgb

# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)
model = None

def load_model():
    global model
    model = load("./lightgbm.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    if flask.request.method == "POST":
        if flask.request.get_json().get("feature"):

            # read feature from json and convert to dataframe
            features = flask.request.get_json().get("feature")
            df_X = pd.DataFrame.from_dict(features)

            # predict
            response["prediction"] = model.predict(df_X).tolist()

            # indicate that the request was a success
            response["success"] = True
    # return the data dictionary as a JSON response
    return flask.jsonify(response)

if __name__ == "__main__":
    load_model()
    print("Server is running ...")
    app.run(host='0.0.0.0', port=5000)
