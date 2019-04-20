import pandas as pd
from sklearn.linear_model import LinearRegression

dataFrame = pd.read_csv('homeprice_OneHotEncoding.csv')
print(dataFrame)

dummyDataFrame = pd.get_dummies(dataFrame.town)
print(dummyDataFrame)

concatinate = pd.concat([dataFrame,dummyDataFrame],axis='columns')
print(concatinate)

finalDataFrame = concatinate.drop(['town','Pune'],axis='columns')
print(finalDataFrame)

model = LinearRegression()
X = finalDataFrame.drop(['price'],axis='columns')
Y = finalDataFrame.price

model.fit(X,Y)
print()
print(model.predict([[3600,0,0]]))
print()
print(model.predict([[3400,0,0]]))

