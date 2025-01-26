import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
y = y.reshape(-1, 1)
X_b = np.c_[np.ones((len(X), 1)), X]  # Add bias term

theta = np.zeros((2, 1))  # Initialize parameters
learning_rate = 0.1
n_iterations = 100

# Gradient Descent
for _ in range(n_iterations):
    theta -= learning_rate * (2 / len(X)) * X_b.T @ (X_b @ theta - y)

# Plot results
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, X_b @ theta, color="red", label="Optimized Line")

# Draw loss lines for each data point
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], X_b[i] @ theta], 'g--', linewidth=1)

plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression With Gradient Descent")
plt.legend()
plt.show()
