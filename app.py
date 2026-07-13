from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("flight_delay_prediction_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    dep_delay = float(request.form["dep_delay"])
    taxi_out = float(request.form["taxi_out"])
    distance = float(request.form["distance"])
    carrier_delay = float(request.form["carrier_delay"])
    weather_delay = float(request.form["weather_delay"])
    nas_delay = float(request.form["nas_delay"])
    security_delay = float(request.form["security_delay"])
    late_aircraft_delay = float(request.form["late_aircraft_delay"])

    sample = np.array([[
        2024,
        7,
        15,
        1,
        245,
        3,
        1025,
        15,
        28,
        5,
        42,
        56,
        8,
        930,
        945,
        dep_delay,
        taxi_out,
        1003,
        1145,
        7,
        1130,
        1152,
        0,
        0,
        0,
        120,
        127,
        102,
        distance,
        carrier_delay,
        weather_delay,
        nas_delay,
        security_delay,
        late_aircraft_delay
    ]])

    prediction = model.predict(sample)

    return render_template(
        "index.html",
        prediction=round(prediction[0],2)
    )


if __name__ == "__main__":
    app.run(debug=True)