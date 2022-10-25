import pandas as pd
import pickle




from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataSet = pd.read_csv("dataset/Salary_Data.csv")
X = dataSet.iloc[:,:-1]
y = dataSet.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/5)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("X_test",X_test)
print("y_pred",y_pred)

fileName = 'SalaryPredictionModel'
pickle.dump(model,open(fileName,'wb'))

loaded_model = pickle.load(open(fileName,'rb'))
loaded_model.predict(X_test)
