import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataSet = pd.read_csv("datasets/50_Startups_WOState.csv")
X = dataSet.iloc[:,:-1].values
y = dataSet.iloc[:,3].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/5, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

# print(y_pred)

X_sample = [[73994.56,122782.75,303319.26]]
y_pred_sample = regressor.predict(X_sample)
print(y_pred_sample)



