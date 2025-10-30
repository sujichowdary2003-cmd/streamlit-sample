from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

scaler = pickle.load(open('scaler.pkl', 'rb'))
lr = pickle.load(open('lr.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/result', methods=['POST'])
def result():
    try:
        Age = int(request.form.get("Age"))
        Glucose = int(request.form.get("Glucose"))
        BloodPressure = int(request.form.get("BloodPressure"))
        Insulin = int(request.form.get("Insulin"))
        BMIs = float(request.form.get("BMI"))
        SkinThickness = int(request.form.get("SkinThickness"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))

        temp_arr = [Glucose, BloodPressure, SkinThickness, Insulin, BMIs, DiabetesPedigreeFunction, Age]
        data = np.array([temp_arr])
        temp_sc = scaler.transform(data)
        pred = int(lr.predict(temp_sc)[0])

        res = "does not indicate" if pred == 0 else "indicates"

    except Exception as e:
        res = f"Error: {str(e)}"

    return render_template('result.html', prediction=res)

if __name__ == '__main__':
    app.run(debug=True)
