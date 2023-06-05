import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

print("Tracing pandas values of Sale (Dollars)")

# Data upload
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('final_Liquor_Sales.csv', header=0)
df["period"] = df["Date"].map(str)
df = df.set_index('period')

aSales = df['Sale (Dollars)'].to_numpy()  # Array of sales data
logdata = np.log(aSales)  # Log transform
data = pd.Series(logdata)  # Convert to pandas series

# Train and test set
train = data[:-12]
test = data[-12:]

# SARIMAX model
order = (1, 0, 1)
seasonal_order = (0, 1, 1, 12)
sarima_model = SARIMAX(train.values, order=order, seasonal_order=seasonal_order)
fit_model = sarima_model.fit()
fit_model.plot_diagnostics(figsize=(10, 8))
plt.show()

# Predictions and forecasts
ypred = fit_model.predict(start=0, end=len(train))
yfore = fit_model.get_forecast(steps=12)
expdata = np.exp(ypred)  # Unlog
expfore = np.exp(yfore.predicted_mean)

plt.plot(expdata)
plt.plot([None for x in expdata] + [x for x in expfore])
plt.show()

# RMSE and MAE
mse = mean_squared_error(np.exp(test), expfore)
mae = mean_absolute_error(np.exp(test), expfore)

print("Root Mean Squared Error (RMSE):", np.sqrt(mse))
print("Mean Absolute Error (MAE):", mae)
