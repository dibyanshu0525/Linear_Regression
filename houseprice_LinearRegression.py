import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv')
print(df.info())
print(df.describe())
print(df.isnull().sum())

X = df.iloc[:,:-2]
y = df.iloc[:,-2]

print(X)
print("____________________")
print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X)
X = imputer.transform(X)
print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred, y_test)))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R-squared of Linear Regression (Coefficient of Determination):", r2)



