import numpy as np
from flask import Flask,request, render_template, jsonify
import pickle
import pandas as pd
import traceback

app = Flask(__name__)
model.predict(open('SalaryPredictionModel.pkl','rb'))

app.route('/getSalary',methods=['GET'])
def getSalary():
    if model:
        try:
            json = request.json
            print(json_["years"])
            final_input = [np.array(json_["years"])]
            print(final_input)
            prediction = model.predict([final_input])
            return jsonify(prediction = str('Salary: {prediction}'))
        except:
            return jsonify(trace = traceback.format_exc())

if __name_ == "__main__":
    app.run(debug=True, port=9091)
