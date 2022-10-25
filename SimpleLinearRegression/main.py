import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from fastapi import FastAPI

app = FastAPI()




@app.get("/main/{years_exp}")
def index(years_exp: float):
    dataSet = pd.read_csv("dataset/Salary_Data.csv")
    X = dataSet.iloc[:,:-1]
    y = dataSet.iloc[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/5)

    model = LinearRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    sample_input = [[years_exp]]

    sample_pred = model.predict(sample_input)

# print("X_test",X_test)
# print("y_pred",y_pred)
# print("sample_pred:",round(sample_pred[0],2))


    return {"value":round(sample_pred[0],2)}

# fileName = 'SalaryPredictionModel'
# pickle.dump(model,open(fileName,'wb'))
#
# loaded_model = pickle.load(open(fileName,'rb'))
# loaded_model.predict(X_test)
