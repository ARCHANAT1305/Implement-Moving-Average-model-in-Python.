# Ex.No: 08 MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date:06/05/25
## AIM:
To implement Moving Average Model and Exponential smoothing Using Python.

## ALGORITHM:
1. Import necessary libraries    
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of the dataset    
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
## PROGRAM:
### NAME : ARCHANA T
### REGISTER NUMBER : 212223240013
```
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import sys

warnings.filterwarnings("ignore")


try:
    data = pd.read_csv('Online Retail.csv', encoding='ISO-8859-1')
except FileNotFoundError:
    print("The dataset 'Online Retail.csv' was not found.")
    sys.exit()


data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], errors='coerce')
data.dropna(subset=['InvoiceDate'], inplace=True)


data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]


data['TotalSales'] = data['Quantity'] * data['UnitPrice']


data.set_index('InvoiceDate', inplace=True)

data = data[(data.index >= '2010-12-01') & (data.index < '2011-12-01')]


power_data = data[['TotalSales']].copy()
power_data.rename(columns={'TotalSales': 'TotalPower'}, inplace=True)


print("Shape of the dataset:", power_data.shape)
print("First 10 rows:")
print(power_data.head(10))

plt.figure(figsize=(12, 6))
plt.plot(power_data['TotalPower'], label='Original Power Consumption')
plt.title('Sales (First 100 Entries)')
plt.xlabel('Date')
plt.ylabel('Total Sales (£)')
plt.legend()
plt.grid()
plt.show()

rolling_mean_5 = power_data['TotalPower'].rolling(window=5).mean()
rolling_mean_10 = power_data['TotalPower'].rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(power_data['TotalPower'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='5-period MA')
plt.plot(rolling_mean_10, label='10-period MA')
plt.title('Moving Averages (First 100 Entries)')
plt.xlabel('Date')
plt.ylabel('Total Sales (£)')
plt.legend()
plt.grid()
plt.show()


monthly_data = power_data.resample('MS').sum()

# Interpolate if missing values exist
if monthly_data.isnull().sum().any():
    print("Missing values found in resampled data. Interpolating...")
    monthly_data.interpolate(method='linear', inplace=True)

# Scaling
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(monthly_data).flatten(),
    index=monthly_data.index
)
scaled_data += 1  # Avoid zero for multiplicative seasonality

split = int(len(scaled_data) * 0.8)
train_data = scaled_data[:split]
test_data = scaled_data[split:]

seasonal_periods = 12
if len(train_data) < 2 * seasonal_periods:
    print(f"Warning: Not enough data for {seasonal_periods} seasonal periods. Reducing.")
    seasonal_periods = min(2, len(train_data))

if len(train_data) < 2 * seasonal_periods or seasonal_periods <= 1:
    print("Insufficient data for seasonality. Using non-seasonal model.")
    seasonal_periods = None


if len(train_data) <= 1:
    print("Error: Not enough data points to fit the model.")
else:
    model = ExponentialSmoothing(
        train_data,
        trend='add',
        seasonal='mul' if seasonal_periods else None,
        seasonal_periods=seasonal_periods
    ).fit()
    forecast = model.forecast(steps=len(test_data))

    plt.figure(figsize=(12, 6))
    train_data.plot(label='Train Data')
    test_data.plot(label='Test Data')
    forecast.plot(label='Forecast')
    plt.title('Sales Forecast (Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Scaled Sales')
    plt.legend()
    plt.grid()
    plt.show()

    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    print("Test RMSE:", rmse)

    final_model = ExponentialSmoothing(
        scaled_data,
        trend='add',
        seasonal='mul' if seasonal_periods else None,
        seasonal_periods=seasonal_periods
    ).fit()
    future_forecast = final_model.forecast(steps=12)

    plt.figure(figsize=(12, 6))
    scaled_data.plot(label='Historical Data')
    future_forecast.plot(label='Future Forecast')
    plt.title('Future Sales Forecast (Next 12 Months)')
    plt.xlabel('Date')
    plt.ylabel('Scaled Sales')
    plt.legend()
    plt.grid()
    plt.show()

 
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/e57257af-81c0-4f26-9f6b-ad59015bb136)
![image](https://github.com/user-attachments/assets/e7d8b5a7-0ae2-4303-a635-6666784956aa)
![image](https://github.com/user-attachments/assets/f4add8fa-e098-45fb-9fbd-e93ec6cd9083)
![image](https://github.com/user-attachments/assets/c6e8c84a-4c33-4ba6-bad3-999f725a662d)
![image](https://github.com/user-attachments/assets/0776239c-3630-4652-8f06-7f7bf7ca1cd5)




## RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
