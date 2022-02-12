#-------------------------------------------------------------------------
# FILENAME: knn_stock.py
# SPECIFICATION: Stock price prediction with KNN regression
# FOR: CS 4210- Group Project
#-----------------------------------------------------------*/
from datetime import date
import ta 
import pandas as pd
import numpy as np
import yfinance as yf
import math
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


#Date for training and testing, test split ratio
end_date = date(2021,7,2)
start_date = date(2018,1,1)
train_date = date(2021,1,6)
SPLIT_PERCENT = 0.86    # Split 86:14 for train and test

#Input the stock ticker
ticker = input("Enter the stock ticker: ")
#Download data with yfinance
train_data = yf.download(ticker, start_date, train_date) #1/3/2017 to 1/5/2021
test_data = yf.download(ticker, train_date, end_date) #1/6/2021 to 7/1/2021

#Print the entire dataset
df = yf.download(ticker, start_date, end_date)
print('\nData of ', ticker, ' from ', start_date, ' to ', end_date, '\n')
print(df.drop(['Open', 'Close', 'Volume'], 1))

#--------------------High and Low-------------------------------------------------
#Match the (High and Low) of today to the (Adj Close) price of tommorow
X_train = train_data.drop(['Open','Close','Volume','Adj Close'],axis = 1) #only keep high and low
X_train.drop(X_train.tail(1).index,inplace=True)   #delete the last day


Y_train = train_data['Adj Close']
Y_train.drop(Y_train.head(1).index,inplace=True)  #delete the first day


x_test = test_data.drop(['Open','Close','Volume','Adj Close'],axis = 1) #only keep high and low
x_test.drop(x_test.tail(1).index,inplace=True)   #Delete the last day

y_test = test_data['Adj Close']
y_test.drop(y_test.head(1).index,inplace=True)   #Delete the first day

#K: from 1 to 25, cv = 5, and Negative MAE scoring
model = GridSearchCV(neighbors.KNeighborsRegressor(), 
                      param_grid =  {'n_neighbors':np.arange(1,25,1),'weights':['distance','uniform'] },
                                    cv=5,  refit=True, scoring= 'neg_mean_absolute_error')
model.fit(X_train, Y_train)
preds_hl = model.predict(x_test)
score_hl = model.score(X_train,Y_train)

#------------------High and EMA-10----------------------------------------------------
#Match the (High and EMA 10) of today to the (Adj Close) price of tommorow
pd.options.mode.chained_assignment = None
df_new = yf.download(ticker, start_date - timedelta(days=16) , end_date)
new_data = df_new[['Adj Close','High']]  #new_data with only adj close
new_data['EMA'] = ta.trend.ema_indicator(new_data['Adj Close'], window = 10) #EMA with period = 10
new_data.dropna(how ='any', inplace =True) #drop all rows with NaN

split = int(SPLIT_PERCENT*len(new_data))

X_train = new_data[:split].drop(['Adj Close'],1) 
X_train.drop(X_train.tail(1).index,inplace=True)   #delete the last day

Y_train = new_data[:split]['Adj Close']
Y_train.drop(Y_train.head(1).index,inplace=True)  #delete the first day

x_test = new_data[split:].drop(['Adj Close'],1) 
x_test .drop(x_test.tail(1).index,inplace=True)   #delete the last day
x_test .drop(x_test.head(1).index,inplace=True)   #Adjust the first day to match with previous testing data

y_test = new_data[split:]['Adj Close']
y_test.drop(y_test.head(2).index,inplace=True)  #delete the first day

model.fit(X_train, Y_train)
preds_h_ema10 = model.predict(x_test)   

#-----------------High and EMA-8------------------------
#Match the (High and EMA 8) of today to the (Adj Close) price of tommorow
df_new = yf.download(ticker, start_date - timedelta(days=12) , end_date)
new_data = df_new[['Adj Close','High']]  #new_data with only adj close
new_data['EMA'] = ta.trend.ema_indicator(new_data['Adj Close'], window = 5) #EMA with period = 5
new_data.dropna(how ='any', inplace =True) #drop all rows with NaN

split = int(SPLIT_PERCENT*len(new_data))

X_train = new_data[:split].drop(['Adj Close'],1) 
X_train.drop(X_train.tail(1).index,inplace=True)   #delete the last day

Y_train = new_data[:split]['Adj Close']
Y_train.drop(Y_train.head(1).index,inplace=True)  #delete the first day

x_test = new_data[split:].drop(['Adj Close'],1) 
x_test .drop(x_test.tail(1).index,inplace=True)   #delete the last day
x_test .drop(x_test.head(1).index,inplace=True) 

y_test = new_data[split:]['Adj Close']
y_test.drop(y_test.head(2).index,inplace=True)  #delete the first day

model.fit(X_train, Y_train)
preds_h_ema5 = model.predict(x_test)   

#Put all predicted of different feature combinations to one dataframe
result = pd.DataFrame({'High-Low': preds_hl,'Open-EMA10': preds_h_ema10, 'Open-EMA5': preds_h_ema5, 'Adj Close': y_test})
print('\nPredicted ', ticker, ' stock price from different feature combinations\n')
print(result)

#Plot the result
plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.ylabel( 'Adj Closing Price')
plt.title(label=ticker, fontsize=30)
plt.plot(test_data['Adj Close'],'b-',label='Adj Closing Price' )
plt.plot(result['High-Low'],'r-',label='Predicted(High-Low)')
plt.legend()
plt.show()
print()

plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.ylabel( 'Adj Closing Price')
plt.title(label=ticker, fontsize=30)
plt.plot(test_data['Adj Close'],'b-',label='Adj Closing Price' )
plt.plot(result['Open-EMA10'],'r-',label='Predicted(High-EMA 10)')
plt.legend()
plt.show()
print()

plt.figure(figsize=(16,8))
plt.xlabel('Date')
plt.ylabel( 'Adj Closing Price')
plt.title(label=ticker, fontsize=30)
plt.plot(test_data['Adj Close'],'b-',label='Adj Closing Price' )
plt.plot(result['Open-EMA5'],'r-',label='Predicted(High-EMA 5)')
plt.legend()
plt.show()
print()


#Printing performance measures
#y_test the same for all feature combinations

print("\nPerformance measure for predicted price of",ticker, '\n')
print("High-Low")
print("MAE:  ", mean_absolute_error(y_test.values,preds_hl))
print("RMSE: ", math.sqrt(mean_squared_error(y_test.values, preds_hl)))
print("R2:   ", r2_score(y_test.values,preds_hl))


print("\nHigh-EMA(10)")
print("MAE:  ", mean_absolute_error(y_test.values,preds_h_ema10))
print("RMSE: ", math.sqrt(mean_squared_error(y_test.values, preds_h_ema10)))
print("R2:   ", r2_score(y_test.values,preds_h_ema10))


print("\nHigh-EMA(5)")
print("MAE:  ", mean_absolute_error(y_test.values,preds_h_ema5))
print("RMSE: ", math.sqrt(mean_squared_error(y_test.values, preds_h_ema5)))
print("R2:   ", r2_score(y_test.values,preds_h_ema5))
