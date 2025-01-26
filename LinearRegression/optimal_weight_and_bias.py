import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
pounds = np.array([3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37])
mpg = np.array([18, 15, 18, 16, 15, 14, 24])

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(pounds, mpg, color='blue', label='Data Points')
plt.xlabel('Pounds in 1000s')
plt.ylabel('Miles per gallon')
plt.title('Scatter Plot of the Data')
plt.legend()
plt.grid()
plt.show()

# Train the linear regression model
model = LinearRegression()
model.fit(pounds.reshape(-1, 1), mpg)

# Get the optimal weight and bias
weight = model.coef_[0]
bias = model.intercept_

print(f"Optimal Weight (Slope): {weight:.2f}")
print(f"Optimal Bias (Intercept): {bias:.2f}")

# Calculate the minimum MSE
y_pred = model.predict(pounds.reshape(-1, 1))
mse = np.mean((mpg - y_pred) ** 2)
print(f"Minimum MSE: {mse:.2f}")

# Plot the linear regression line
plt.figure(figsize=(8, 6))
plt.scatter(pounds, mpg, color='blue', label='Data Points')
plt.plot(pounds, y_pred, color='red', label='Linear Regression Line')
plt.xlabel('Pounds in 1000s')
plt.ylabel('Miles per gallon')
plt.title('Linear Regression Model')
plt.legend()
plt.grid()
plt.show()
