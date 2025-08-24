# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 23:42:14 2023

@author: caubu
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib

# # Load data from xlsx file into a pandas DataFrame
# df = pd.read_excel("evergreen_broadleaf_forest_ML2.xlsx")
# df=df.drop(['Count', 'TimeSJ', 'Lucc', 'Cx', 'Cy', 'year', 'month', 'day', 'hour', 'Time', 'Time_day'], axis=1)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df.drop('MEAN', axis=1), df['MEAN'], test_size=0.3)

X_train = pd.read_excel("X_train.xlsx")
X_test = pd.read_excel("X_test.xlsx")
y_train = pd.read_excel("y_train.xlsx")
y_test = pd.read_excel("y_test.xlsx")

# Train a random forest regressor on the training data
regressor = RandomForestRegressor()
regressor.fit(X_train, y_train["MEAN"])

# Use the trained model to make predictions on the testing data
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("r2:", r2)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print("rmse:", rmse)

plt.plot(y_test["MEAN"], y_pred,'o')

# Save the trained model to a file
joblib.dump(regressor, "RF.joblib")

my_dataframe = pd.DataFrame({"y_test": y_test["MEAN"], "y_pred": y_pred})
my_dataframe.to_excel('RF.xlsx', index=False)
