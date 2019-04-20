import pandas as pd
import numpy as np
from sklearn import linear_model
import math


def from_sklearn():
    df = pd.read_csv('test_scores.csv')
    regression = linear_model.LinearRegression()
    regression.fit(df[['math']], df.cs)
    return regression.coef_, regression.intercept_


def from_gradient_decent(x, y):
    m_curr = b_curr = 0
    iteration = 1000000
    n = len(x)
    learning_rate = 0.0002
    cost_previous = 0
    for i in range(iteration):
        y_prediction = m_curr * x + b_curr
        cost = (1 / n) * sum([val ** 2 for val in (y - y_prediction)])
        md = -(2 / n) * sum(x * (y - y_prediction))
        bd = -(2 / n) * sum(y - y_prediction)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print("M = {}, B = {} , Cost = {}, Iterations = {}".format(m_curr, b_curr, cost, iteration))

    return m_curr, b_curr


if __name__ == '__main__':
    df = pd.read_csv('test_scores.csv')
    x = np.array(df.math)
    y = np.array(df.cs)
    print(x)
    print(y)
    m, b = from_gradient_decent(x, y)
    print("Using Gradient Decent Algorithm we got the value as M = {} , B = {}".format(m, b))

    mSklearn, bSklearn = from_sklearn()
    print("Using Sklearn M = {} and B = {}".format(mSklearn, bSklearn))
    # Using Sklearn M = [1.01773624] and B = 1.9152193111569034
# Using Gradient Decent Algorithm we got the value as M = 1.0177381667350405 , B = 1.9150826165722297
# Using Sklearn M = [1.01773624] and B = 1.9152193111569034

