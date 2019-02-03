import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('dataset.csv')

X = dataset['Head Size(cm^3)'].values
Y = dataset['Brain Weight(grams)'].values

print('Scratch Python Implementation!!!')

x_mean = np.mean(X)
y_mean = np.mean(Y)

n = len(X)

numerator = 0
denominator = 0
for i in range(n):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i] - x_mean) ** 2
    
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

max_x = np.max(X) + 100
min_x = np.min(X) - 100

x = np.linspace(min_x, max_x, 10)
y = b0 + b1 * x

plt.scatter(X, Y, c='g', label='Scatter Plot')
plt.plot(x, y, color='b', label='Regression Line')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

rmse = 0
for i in range(n):
    y_pred=  b0 + b1* X[i]
    rmse += (Y[i] - y_pred) ** 2
    
rmse = np.sqrt(rmse/n)
print(rmse)

ss_t = 0
ss_r = 0
for i in range(n) :
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - y_mean) ** 2
    ss_r += (Y[i] - y_pred) **2
    
score  = 1 - (ss_r/ss_t)
print(score)

print('Scikit-Learn Implementation!!!')

X = X.reshape((n, 1))

lin_reg = LinearRegression()
lin_reg = lin_reg.fit(X, Y)

Y_pred = lin_reg.predict(X)

mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
print(rmse)

r2_score = lin_reg.score(X, Y)
print(r2_score)