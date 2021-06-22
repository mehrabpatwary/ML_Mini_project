import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = (20.0, 10.0)

# Reading Data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("headbrain.csv")
print(data.shape)
print(data.head())

# Collecting X and Y

X = data['Head Size(cm^3)'].values
Y = data["Brain Weight(grams)"].values

# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

# Total number of values
n = len(X)

# Using the formula to calculate b1 and b0
numer = 0
denom = 0

for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)

# print coefficients
print(m, c)

# Plotting Value and Regression Line
max_x = np.max(X) + 100
min_x = np.min(X) - 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = c + m * x

# ploting line
plt.plot(x, y, color='#58b970', label='Regression Line')
# ploting Scatter polints
plt.scatter(X, Y, c="#ef5423", label="Scatter Plot")

plt.xlabel("Head Size in cm3")
plt.ylabel("Brain Weight in grams")
plt.legend()
plt.show()

ss_t = 0
ss_r = 0
y_pred = []
for i in range(n):
    y_pred.append(((m * X[i]) + c))
    ss_r += (((m * X[i]) + c) - mean_y) ** 2
    ss_t += (Y[i] - mean_y) ** 2
r2 = (ss_r / ss_t)
print(r2)

print(r2_score(Y, y_pred=y_pred))
