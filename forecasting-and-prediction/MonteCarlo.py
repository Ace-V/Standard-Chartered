import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf  

def get_data(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end) 
    stockData = stockData['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

stockList = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
stocks = stockList
enddate = dt.datetime.now()
startdate = enddate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startdate, enddate)

weights=np.random.random(len(meanReturns))
weights /= np.sum(weights)

mc_sims =100
T=100
meanM=np.full(shape=(T,len(weights)),fill_value=meanReturns)
meanM=meanM.T

portfolio_sims=np.full(shape=(T,mc_sims),fill_value=0.0)
intialportfolio=10000

for m in range(0,mc_sims):
    Z=np.random.normal(size=(T,len(weights)))
    L=np.linalg.cholesky(covMatrix)
    dailyreturns=meanM+np.inner(L,Z)
    portfolio_sims[:,m]= np.cumprod(np.inner(weights,dailyreturns.T)+1)*intialportfolio

    plt.plot(portfolio_sims)
    plt.ylabel('Portfolia Value($)')
    plt.xlabel('Days')
    plt.title('MC Simulations of a stock portfolio')
    plt.show()

