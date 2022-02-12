from datetime import date
import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# create pandas dataframe with info from yahoo finance
#df = data.DataReader('MSFT', 'yahoo', start_date, end_date)
df = pd.read_csv('AAPL training data.csv', sep=',')

predict_df = pd.read_csv('AAPL prediction data.csv', sep=',')

# Keep only the 'Adj Close' 
# Adjusted close is a more accurate evaluation of the stock price
# factors in dividends, stock splits, new stock offerings, etc. 
df = df[['Adj Close']]
predict_df = predict_df[['Adj Close']]

adj_closing = []
for i in predict_df['Adj Close']:
    adj_closing.append(i)

for EMA in range(8, 13):
    dframe = df.copy()
    p_df = predict_df.copy()
    
    # Calculate the Exponential Moving Average using Technical Analysis module
    dframe['EMA'] = ta.trend.EMAIndicator(dframe['Adj Close'], window = EMA).ema_indicator()
    
    p_df['EMA'] = ta.trend.EMAIndicator(p_df['Adj Close'], window = EMA).ema_indicator()
    
    # Drop the first X rows with NaN values where X = EMA
    dframe = dframe.iloc[EMA:]
    p_df = p_df.iloc[EMA:]
    
    selected_columns = p_df[['Adj Close']]
    adj_c = selected_columns.copy()
    
    p_df.drop(['Adj Close'], axis=1, inplace=True)
    
    # The goal is to use the Exponential Moving Average to predict the adjusted closing value
    Y_train, Y_test, X_train, X_test = train_test_split(dframe[['Adj Close']], dframe[['EMA']], test_size=0.25, shuffle = False)
    
    # create and train model
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_test_pred = model.predict(X_test)
    Y_pred = model.predict(p_df)
    
    pred_score = model.score(X_test,Y_test)
    
    print('EMA:', EMA)
    #print('Prediction Score :', pred_score)
    #print("Model Coefficients:", model.coef_[0][0])
    print("Mean Absolute Error Training Data:", mean_absolute_error(Y_test, Y_test_pred))
    print("Mean Absolute Error Test Data:", mean_absolute_error(adj_c, Y_pred))
    print()
    print("Mean Squared Error Training Data:", mean_squared_error(Y_test, Y_test_pred))
    print("Mean Squared Error Test Data:", mean_squared_error(adj_c, Y_pred))
    print()
    print("Coefficient of Determination Training Data:", r2_score(Y_test, Y_test_pred))
    print("Coefficient of Determination Test Data:", r2_score(adj_c, Y_pred))
    print("\n")
    
    Y_pred_values = list(dframe['Adj Close'])
    offset = len(dframe) - len(Y_test_pred)
    for i in range(len(Y_test_pred)):
        Y_pred_values[offset + i] = Y_test_pred[i]
    X = offset
        
    overall_graph = adj_closing.copy()
    offset = len(adj_closing) - len(Y_pred)
    for i in range(len(Y_pred)):
        adj_closing[offset + i] = Y_pred[i][0]
    
    #plt.plot(list(dframe['Adj Close']), label = "Adj Close")
    #plt.plot(Y_pred_values, label = "Training Predictions")
    #plt.axvline(X)
    title = 'AAPL Stock Price Trained w/ EMA ' + str(EMA)
    plt.title(title)
    plt.xlabel('Trading Days')
    plt.ylabel('Adj Closing Price')
    
    plt.plot(overall_graph, label = "Adj Close")
    plt.plot(adj_closing, label = "Predictions")
    
    plt.legend()
    
    plt.show()
