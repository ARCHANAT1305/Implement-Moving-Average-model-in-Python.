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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

# Load dataset
try:
    data = pd.read_csv('powerconsumption.csv')
except FileNotFoundError:
    print("The dataset 'powerconsumption.csv' was not found.")
    exit()

# Convert 'Datetime' to datetime format
data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
data.dropna(subset=['Datetime'], inplace=True)
data.set_index('Datetime', inplace=True)

# Limit to first 100 rows
data = data.iloc[:100]

# Combine power consumption across all zones
data['TotalPower'] = (
    data['PowerConsumption_Zone1'] +
    data['PowerConsumption_Zone2'] +
    data['PowerConsumption_Zone3']
)

# Use only TotalPower
power_data = data[['TotalPower']]

# Display basic info
print("Shape of the dataset:", power_data.shape)
print("First 10 rows:")
print(power_data.head(10))

# Plot original power consumption
plt.figure(figsize=(12, 6))
plt.plot(power_data['TotalPower'], label='Original Power Consumption')
plt.title('Power Consumption (First 100 Entries)')
plt.xlabel('Date')
plt.ylabel('Power (kWh)')
plt.legend()
plt.grid()
plt.show()

# Moving averages
rolling_mean_5 = power_data['TotalPower'].rolling(window=5).mean()
rolling_mean_10 = power_data['TotalPower'].rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(power_data['TotalPower'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='5-period MA')
plt.plot(rolling_mean_10, label='10-period MA')
plt.title('Moving Averages (First 100 Entries)')
plt.xlabel('Date')
plt.ylabel('Power (kWh)')
plt.legend()
plt.grid()
plt.show()

# Resample to monthly totals (may result in fewer than expected points)
monthly_data = power_data.resample('MS').sum()

if monthly_data.isnull().sum().any():
    print("Missing values found in resampled data. Interpolating...")
    monthly_data.interpolate(method='linear', inplace=True)

# Scaling
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(monthly_data).flatten(),
    index=monthly_data.index
)
scaled_data += 1  # Avoid zero for multiplicative

# Train-test split
split = int(len(scaled_data) * 0.8)
train_data = scaled_data[:split]
test_data = scaled_data[split:]

# Determine seasonal_periods
seasonal_periods = 12
if len(train_data) < 2 * seasonal_periods:
    print(f"Warning: Not enough data for {seasonal_periods} seasonal periods. Reducing.")
    seasonal_periods = min(2, len(train_data))

if len(train_data) < 2 * seasonal_periods or seasonal_periods <= 1:
    print("Insufficient data for seasonality. Using non-seasonal model.")
    seasonal_periods = None

# Fit and forecast
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
    plt.title('Power Consumption Forecast (Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Scaled Power')
    plt.legend()
    plt.grid()
    plt.show()

    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    print("Test RMSE:", rmse)

    # Final forecast
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
    plt.title('Future Forecast (Next 12 Months)')
    plt.xlabel('Date')
    plt.ylabel('Scaled Power')
    plt.legend()
    plt.grid()
    plt.show()
 
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/11cde568-bb17-4905-9797-2eb21f730e0e)


![image](https://github.com/user-attachments/assets/bcfaa126-d17e-4641-a221-9284a2610b90)

![image](https://github.com/user-attachments/assets/80baaa12-10bc-4f7c-bd92-7c46d5841485)



## RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
