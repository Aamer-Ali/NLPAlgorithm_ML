import pandas as pd
import numpy as np
import math
from sklearn import linear_model
from word2number import w2n

# Reading File
df = pd.read_csv('hiring.csv')
print(df)
print()

# Filling data with zero
df.experience = df.experience.fillna('zero')
print(df)
print()

# Converting Words to number
df.experience = df.experience.apply(w2n.word_to_num)
print(df)
print()

mean_testScores = math.floor(df['test_score(out of 10)'].mean())
print(mean_testScores)
print()

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(mean_testScores)
print(df)
print()

reg  = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])
print("Prediction :")
print(reg.predict([[2,9,6]]))
print(reg.predict([[12,10,10]]))