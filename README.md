# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### Developed by: POZHILAN V D
### Register no: 212223240118


### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('BMW_Car_Sales_Classification.csv')

# Display column names to identify the correct column for grouping
print(data.columns)

# Group by Year and calculate average Price_USD per year
resampled_data = data.groupby('Year')['Price_USD'].mean().reset_index()

# Extract years and price values
years = resampled_data['Year'].tolist()
price = resampled_data['Price_USD'].tolist()

# Center X values around the middle year
X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, price)]
n = len(years)

# Linear regression calculation
b = (n * sum(xy) - sum(price) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(price) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

# Polynomial regression (2nd degree)
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, price)]
coeff = [[len(X), sum(X), sum(x2)], [sum(X), sum(x2), sum(x3)], [sum(x2), sum(x3), sum(x4)]]
Y = [sum(price), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

# Print equations
print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"Polynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

# Add trends to dataframe
resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend

# Plot Linear Trend
plt.figure(figsize=(10,6))
plt.plot(resampled_data['Year'], resampled_data['Price_USD'], 'bo-', label='Average Price USD')
plt.plot(resampled_data['Year'], resampled_data['Linear Trend'], 'k--', label='Linear Trend')
plt.xlabel('Year')
plt.ylabel('Average Price USD')
plt.title('Linear Trend of Average Price USD over Years')
plt.legend()
plt.grid(True)
plt.show()

# Plot Polynomial Trend
plt.figure(figsize=(10,6))
plt.plot(resampled_data['Year'], resampled_data['Price_USD'], 'bo-', label='Average Price USD')
plt.plot(resampled_data['Year'], resampled_data['Polynomial Trend'], 'r-', label='Polynomial Trend')
plt.xlabel('Year')
plt.ylabel('Average Price USD')
plt.title('Polynomial Trend of Average Price USD over Years')
plt.legend()
plt.grid(True)
plt.show()


```

### OUTPUT


A - LINEAR TREND ESTIMATION
<img width="892" height="639" alt="image" src="https://github.com/user-attachments/assets/ff3b34d7-fac7-4a1a-97a3-0fbd7d7f3ebf" />



B- POLYNOMIAL TREND ESTIMATION
<img width="909" height="541" alt="image" src="https://github.com/user-attachments/assets/5cd8dec7-0421-4bc8-8d12-a571a22829e8" />



### RESULT:
Thus ,the python program for linear and Polynomial Trend Estimation has been executed successfully.
