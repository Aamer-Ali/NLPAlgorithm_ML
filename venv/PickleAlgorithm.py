import pandas as pd
import numpy as mp
from sklearn.linear_model import LinearRegression

dataFrame = pd.read_csv('homeprices.csv')
print(dataFrame)

linearRegression = LinearRegression()
linearRegression.fit(dataFrame[['area']], dataFrame.price)
print(linearRegression.predict([[2600]]))


with open('pickleLearning', 'wb') as mFile:
    pickle.dump(linearRegression, mFile)

with open('pickleLearning', 'rb') as mFile:
    mp = pickle.load(mFile)

print(mp.predict([[2600]]))