import pandas as pd
import numpy as np
from sklearn import linear_model
import math

# reading the csv file for the training purpose
df = pd.read_csv('homeprice.csv')
print("Our Data Frame")
print(df)
print()

# process to fill the missing data from the training table
median_bedroom =  math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedroom)
print("Filling Data frames with values")
print(df)
print()

# Regression
reg =linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

print("Coefficient : ")
print(reg.coef_)
print()
print("Intercept : ")
print(reg.intercept_)
print()
print("Prediction : ")
print(reg.predict([[3000,3,40]]))
print()

# Formula we are using is y = m1x1+m2x2+m3x3+b
# --- m1,m2,m3 are Coefficient
# --- b is intercept
# firstCoef*FirstValue+Secondcoef*SecondValue+ThirdCoef*ThirdValue+Intercept
print("Using Formula m1x1+m2x2+m3x3+b : ")
print(137.25*3000+-26025*3+-6825*40+383724.9999999998)
