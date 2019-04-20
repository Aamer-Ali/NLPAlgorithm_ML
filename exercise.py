import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
from sklearn import linear_model

df = pd.read_csv('exercise.csv')
print(df.head(4))


reg =  linear_model.LinearRegression()
reg.fit(df[['year']],df.perCapitaIncome)

print("Predict Income in year year 2020 is : "+ str(reg.predict([[2020]])))