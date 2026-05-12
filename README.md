# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 30/04/2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

``
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_excel('/content/supermarket_10years.xlsx')

data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

N = len(data)

plt.rcParams['figure.figsize'] = [12, 6]

X = data['Sales_Amount']

# Original Data Plot
plt.plot(X)
plt.title('Original Data (Sales Amount)')
plt.show()

# ACF & PACF
plt.subplot(2, 1, 1)
plot_acf(X, lags=10, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=10, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# ARMA(1,1)
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, N])
plt.show()

plot_acf(ARMA_1)
plt.title('ACF of Simulated ARMA(1,1)')
plt.show()

plot_pacf(ARMA_1)
plt.title('PACF of Simulated ARMA(1,1)')
plt.show()

# ARMA(2,2)
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 5)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, N * 5])
plt.show()

plot_acf(ARMA_2)
plt.title('ACF of Simulated ARMA(2,2)')
plt.show()

plot_pacf(ARMA_2)
plt.title('PACF of Simulated ARMA(2,2)')
plt.show()
```

OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="1307" height="662" alt="image" src="https://github.com/user-attachments/assets/2188fbb6-83ba-4b36-8932-a41e7d20d4a6" />

Partial Autocorrelation:
<img width="1432" height="724" alt="image" src="https://github.com/user-attachments/assets/1a0d6e32-8f31-4a8c-8ec1-d245dfe2afdb" />

Autocorrelation
<img width="1297" height="663" alt="image" src="https://github.com/user-attachments/assets/20e69c26-7887-447a-ab0f-bbde79865b1b" />

SIMULATED ARMA(2,2) PROCESS:
<img width="1261" height="658" alt="image" src="https://github.com/user-attachments/assets/66896385-7472-4eb3-af34-9e93ee297f50" />

Partial Autocorrelation
<img width="1269" height="672" alt="image" src="https://github.com/user-attachments/assets/bd4b01ce-f3e5-42c3-acca-664a13eb3bd1" />

Autocorrelation
<img width="1237" height="646" alt="image" src="https://github.com/user-attachments/assets/3ecd91d5-3869-4410-88c1-172313566113" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
