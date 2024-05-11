#Estimation Time Series Decomposition (ETS)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from pmdarima import auto_arima
dataset=pd.read_csv("AirPassengers.csv",index_col='Month',parse_dates=True)
result=seasonal_decompose(dataset['#Passengers'],model='multiplicative')
result.plot()
plt.show()
print(dataset)
result=auto_arima(dataset['#Passengers'],start_p=1,start_q=1,max_p=3,max_q=3,m=12,start_P=0,seasonal=True,d=None,D=1,trace=True,stepwise=True)

# result.summary()
# result.plot()
# plt.show()
# print(dataset)

#SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
train=dataset.iloc[:len(dataset)-12]
test=dataset.iloc[len(dataset)-12:]
model=SARIMAX(train['#Passengers'],order=(0,1,1),seasonal_order=(2,1,1,12))
result=model.fit()
result.summary()
start=len(train)
end=len(train)+len(test)-1
pred=result.predict(start,end,typ='levels').rename('prediction')
pred.plot(legend=True)
test['#Passengers'].plot(legend=True)
plt.show()