import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df  = pd.read_csv("homeprices.csv")
plt.xlabel('area(sqr ft)')
plt.ylabel('price(INR)')
plt.scatter(df.area,df.price,color='red',marker='+')
plt.show()

# Linear Regression
# create linear Regression object

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# Prediction
print(reg.predict([[3300]]))

d = pd.read_csv("areas.csv")
print(d.head(3))

p = reg.predict(d)
d['price'] = p
d.to_csv('prediction.csv')

plt.xlabel('area(sqr ft)',fontsize=20)
plt.ylabel('price(INR)',fontsize=20)
plt.scatter(df.area,df.price,color='red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.show()

