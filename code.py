```# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 21:38:34 2021

@author: SHRIDHAR KAPSE
"""

#importing modules 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import statsmodels.api as sm
import math
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

#Question_1
# a
df = pd.read_csv("daily_covid_cases.csv")
x=[]
for i in range(612):
    x.append(i)
y= df['new_cases']

plt.plot(x,y)
plt.xlabel('Month-Year')
plt.ylabel('New Confirmed Cases')
plt.show()

# b
#converting a series into list 
y_L_0 = df['new_cases'].tolist()
#inserting 0 at places where after time lag remains empty
y_L_0.insert(612,0)
y_L_1 = df['new_cases'].tolist()
y_L_1.insert(0,0)
corr, _ = pearsonr(y_L_0, y_L_1)
corr

# c
# scatter plot between lag of 1 and original
plt.scatter(y_L_0, y_L_1, c ="blue",linewidths=0.5)
  
# To show the plot
plt.show()

# d
y_L_2=y_L_1.copy()
y_L_2.insert(0,0)

y_L_3=y_L_2.copy()
y_L_3.insert(0,0)

y_L_4=y_L_3.copy()
y_L_4.insert(0,0)

y_L_5=y_L_4.copy()
y_L_5.insert(0,0)

y_L_6=y_L_5.copy()
y_L_6.insert(0,0)

y_L_0.insert(613,0)

corr2, _ = pearsonr(y_L_0, y_L_2)
y_L_0.insert(614,0)
corr3, _ = pearsonr(y_L_0, y_L_3)
y_L_0.insert(615,0)
corr4, _ = pearsonr(y_L_0, y_L_4)
y_L_0.insert(616,0)
corr5, _ = pearsonr(y_L_0, y_L_5)
y_L_0.insert(617,0)
corr6, _ = pearsonr(y_L_0, y_L_6)

print("The correlation between lag 2 sequence and original is : ",corr2)
print("The correlation between lag 3 sequence and original is : ",corr3)
print("The correlation between lag 4 sequence and original is : ",corr4)
print("The correlation between lag 5 sequence and original is : ",corr5)
print("The correlation between lag 6 sequence and original is : ",corr6)


 # e
lag=[1,2,3,4,5,6]
sm.graphics.tsa.plot_acf(y)
plt.xlabel('Lagged Values')
plt.ylabel('Correlation coefficients')
plt.show()

# Question 2
#dividing the data into train part and test part 
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
test_size = 0.35 # 35% for testing
X = series.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# a
L=5
#training the model
model = AR(train, lags=L)
# fit/train the model
model_fit = model.fit() 
# Get the coefficients of AR model
coef = model_fit.params 
#printing the coefficients
print('The coefficients obtained from the AR model are', coef)

# b
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-L:]
history = [history[i] for i in range(len(history))]

predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-L,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(L):
        yhat += coef[d+1] * lag[L-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.
# ii)
plt.plot(test,predictions, color='blue')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()

# b
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-L:]
history = [history[i] for i in range(len(history))]

predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-L,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(L):
        yhat += coef[d+1] * lag[L-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.
# ii)
plt.plot(test,predictions, color='blue')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()

# i)
plt.scatter(test,predictions, color='blue',s=10)
plt.xlabel('actual')
plt.ylabel('predicted')

# iii)
# computing rmse
rmse_per = (math.sqrt(mean_squared_error(test, predictions))/np.mean(test))*100
print('RMSE(%):',rmse_per)

# computing MAPE
mape = np.mean(np.abs((test - predictions)/test))*100
print('MAPE:',mape)

# Question 3
lag_val = [1,5,10,15,25]
RMSE = []
MAPE = []
for l in lag_val:
    model = AutoReg(train, lags=l)
  # fit/train the model
    model_fit = model.fit()
    coef = model_fit.params 
    history = train[len(train)-l:]
    history = [history[i] for i in range(len(history))]
    predicted = list() # List to hold the predictions, 1 step at a time for t in range(len(test)):
    for t in range(len(test)):
        length = len(history)
        Lag = [history[i] for i in range(length-l,length)] 
        yhat = coef[0] # Initialize to w0
        for d in range(l):
             yhat += coef[d+1] * Lag[l-d-1] # Add other values 
        obs = test[t]
        predicted.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.

    # computing rmse
    rmse_per = (math.sqrt(mean_squared_error(test, predicted))/np.mean(test))*100
    RMSE.append(rmse_per)

    # computing MAPE
    mape = np.mean(np.abs((test - predicted)/test))*100
    MAPE.append(mape)

# RMSE (%) and MAPE between predicted and original data values wrt lags in time sequence
data = {'Lag value':lag_val,'RMSE(%)':RMSE, 'MAPE' :MAPE}
print('Table 1\n',pd.DataFrame(data))

# plotting RMSE(%) vs. time lag
plt.xlabel('Time Lag')
plt.ylabel('RMSE(%)')
plt.title('RMSE(%) vs. time lag')
plt.xticks([1,2,3,4,5],lag_val)
plt.bar([1,2,3,4,5],RMSE)
plt.show()

# plotting MAPE vs. time lag
plt.xlabel('Time Lag')
plt.ylabel('MAPE')
plt.title('MAPE vs. time lag')
plt.xticks([1,2,3,4,5],lag_val)
plt.bar([1,2,3,4,5],MAPE)
plt.show()


# Question 4
# computing number of optimal value of p
p = 1
while p < len(df):
  corr = pearsonr(train[p:].ravel(), train[:len(train)-p].ravel())
  if(abs(corr[0]) <= 2/math.sqrt(len(train[p:]))):
    print('The heuristic value for the optimal number of lags is',p-1)
    break
  p+=1

p=p-1
# training the model
model = AutoReg(train, lags=p)
# fit/train the model
model_fit = model.fit()
coef = model_fit.params 
history = train[len(train)-p:]
history = [history[i] for i in range(len(history))]
predicted = list() # List to hold the predictions, 1 step at a time for t in range(len(test)):
for t in range(len(test)):
    length = len(history)
    Lag = [history[i] for i in range(length-p,length)] 
    yhat = coef[0] # Initialize to w0
    for d in range(p):
        yhat += coef[d+1] * Lag[p-d-1] # Add other values 
    obs = test[t]
    predicted.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

# computing rmse
rmse_per = (math.sqrt(mean_squared_error(test, predicted))/np.mean(test))*100
print('RMSE(%):',rmse_per)

# computing MAPE
mape = np.mean(np.abs((test - predicted)/test))*100
print('MAPE:',mape)