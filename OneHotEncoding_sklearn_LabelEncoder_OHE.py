import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataFrame = pd.read_csv('homeprice_OneHotEncoding.csv')
print(dataFrame)
print()
le = LabelEncoder()

dfle = dataFrame
dfle.town = le.fit_transform(dfle.town)
print(dfle)
print()

X = dfle[['town', 'area']].values
print(X)
print()

Y = dfle.price
print(Y)
print()

ohe = OneHotEncoder(categorical_features=[0])

X = ohe.fit_transform(X).toarray()
print(X)
print()

X = X[:, 1:]
print(X)
print()

model = LinearRegression()
model.fit(X, Y)

print(model.predict([[0, 0, 2600]]))
