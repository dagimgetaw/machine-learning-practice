import numpy as np
from sklearn.linear_model import LinearRegression

weights = np.array([3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37]).reshape(-1, 1)
mpg = np.array([18, 15, 18, 16, 15, 14, 24])

model = LinearRegression()
model.fit(weights, mpg)

slope = model.coef_[0]
bias = model.intercept_

print(f"Weight = {slope}")
print(f"Bias = {bias}")