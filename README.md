# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 
### Name: Koduru Sanath Kumar Reddy 
### Register no: 212221240024

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
~~~
# Import necessary libraries
!pip install pmdarima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Electric_Production.csv", index_col='DATE', parse_dates=True)
data = data.asfreq('MS')  # Set frequency to monthly, adjust if needed
data.plot(figsize=(10, 5))
plt.title("Electric Production Over Time")
plt.ylabel("Electric Production")
plt.xlabel("Date")
plt.show()

def adf_test(series):
    result = adfuller(series, autolag='AIC') # Use autolag to automatically select optimal lag
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    return result[1] < 0.05  # True if stationary

is_stationary = adf_test(data["IPG2211A2N"])

plot_acf(data["IPG2211A2N"], lags=20)
plt.show()

plot_pacf(data["IPG2211A2N"], lags=20)
plt.show()

p, d, q = 1, 1, 1       # ARIMA terms
P, D, Q, S = 1, 1, 1, 12 # Seasonal terms, with seasonality cycle S=12 (monthly data)

model = SARIMAX(data["IPG2211A2N"], order=(p, d, q), seasonal_order=(P, D, Q, S))
fitted_model = model.fit()
print(fitted_model.summary())

forecast_steps = 12  # Forecast for the next 12 months
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_index = pd.date_range(data.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')

plt.plot(data, label="Historical Data")
plt.plot(forecast_index, forecast, label="Forecast", color='orange')
plt.legend()
plt.title("Electric Production Forecast")
plt.show()

from pmdarima import auto_arima

auto_model = auto_arima(data["IPG2211A2N"], seasonal=True, m=12)  # m=12 for monthly seasonality
print(auto_model.summary())

train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

model = SARIMAX(train_data["IPG2211A2N"], order=(p, d, q), seasonal_order=(P, D, Q, S))
fitted_train_model = model.fit()

start = len(train_data)
end = len(data) - 1  
predictions = fitted_train_model.predict(start=start, end=end) # Specify end

rmse = np.sqrt(mean_squared_error(test_data["IPG2211A2N"], predictions)) # Access column
print("RMSE:", rmse)
~~~

### OUTPUT:
### Electric Production over time
<img width="671" alt="image" src="https://github.com/user-attachments/assets/987ae7f7-bd7a-4f66-81ae-9e4db4d5e204">

### Autocorrelation
<img width="469" alt="image" src="https://github.com/user-attachments/assets/5caf4afd-d960-4f2f-8d38-72987d1ee81a">


### Partial Autocorrelation
<img width="469" alt="image" src="https://github.com/user-attachments/assets/4bea2be1-aa03-4ca6-877f-5c3b74827637">

### Model Results
<img width="657" alt="image" src="https://github.com/user-attachments/assets/1af6d483-cd1b-427b-8a60-1e56904f5bdf">

### Model Forecast
![Uploading image.png…]()



### RESULT:
Thus the program run successfully based on the SARIMA model.
