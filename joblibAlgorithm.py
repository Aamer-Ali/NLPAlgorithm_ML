import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

dataFrame = pd.read_csv('homeprices.csv')
print(dataFrame)

linearRegression = LinearRegression()
linearRegression.fit(dataFrame[['area']], dataFrame.price)
print()
print("By Linear Regression algorithm = ")
print(linearRegression.predict([[2600]]))

print()
print("By joblib algorithm = ")
joblib.dump(linearRegression, 'joblibLearning')
mp = joblib.load('joblibLearning')
print(mp.predict([[2600]]))
