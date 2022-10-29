#import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load customer data

dataset_file1 = pd.read_csv("../dataset/Data1.csv")
dataset_file2 = pd.read_csv("../dataset/Data2.csv")

#size of data

#print(dataset_file1.shape)
#print(dataset_file2.shape)

#merge the data frames

customer_data = dataset_file1.merge(dataset_file2, how='inner', on='ID')

#size of new customer_data

#print(customer_data.shape)

#get data types

#print(customer_data.dtypes)

#data description - count, mean, IQR, max

#print(customer_data.describe().transpose())


#dropping columns that are irrelevant

customer_data = customer_data.drop(columns='ID')
#print(customer_data.shape)

#check null values
#print(customer_data.isnull().sum())

#remove null values from LoanOnCard as it can't be replaced with mean or mode. No need for imputation

customer_data = customer_data.dropna()
print(customer_data.shape)

# drop highly correlated data

customer_data = customer_data.drop(columns='Age')
#train split

X = customer_data.drop('LoanOnCard', axis=1) #remove predictor feature column
Y = customer_data['LoanOnCard'] #predictor feature column

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=1) # 1 is a random seed number










