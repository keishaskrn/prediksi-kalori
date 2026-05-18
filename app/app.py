from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None

    if request.method == 'POST':

        age = float(request.form['age'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])

        data = np.array([
            [age, weight, duration]
        ])

        data = scaler.transform(data)

        prediction = model.predict(data)[0]

    return render_template(
        'index.html',
        prediction=prediction
    )

if __name__ == '__main__':
    app.run(debug=True)
