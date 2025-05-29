import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('house.csv')

# Extract feature and target
X = data['Area'].values
y = data['Price'].values

# Manually split into training and testing sets (80% train, 20% test)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Simple Linear Regression using Least Squares
# y = mX + c --> m = slope, c = intercept

# Mean of X and y
mean_x = np.mean(X_train)
mean_y = np.mean(y_train)

# Calculate slope (m) and intercept (c)
numerator = np.sum((X_train - mean_x) * (y_train - mean_y))
denominator = np.sum((X_train - mean_x) ** 2)
slope = numerator / denominator
intercept = mean_y - slope * mean_x

print("Slope (Coefficient):", slope)
print("Intercept:", intercept)

# Predict values using the line equation
y_pred = slope * X_test + intercept

# Evaluation Metrics
mae = np.mean(np.abs(y_test - y_pred))
mse = np.mean((y_test - y_pred) ** 2)
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_res = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (ss_res / ss_total)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Plotting
plt.scatter(X_test, y_test, color='blue', label='Actual Price')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price - Linear Regression (Manual)')
plt.legend()
plt.grid(True)
plt.show()
