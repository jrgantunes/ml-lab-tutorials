# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:14:05 2019

@author: user
"""
#%%
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import re
import requests as r
from bs4 import BeautifulSoup as soup
import datetime as dt
import math

#%%


def get_risk(prices):
    return (prices / prices.shift(1) - 1).dropna().std().values

def get_return(prices):
    return ((prices / prices.shift(1) - 1).dropna().mean() * np.sqrt(250)).values

#%%
# Create a function that get a "stock" and "start_date" and "end_date"
# Example of a URL
#url = 'https://finance.yahoo.com/quote/IWD/history?period1=1546300800&period2=1561071600&interval=1d&filter=history&frequency=1d'
def get_url(stock,start_date, end_date):
    # Handle Dates 
    # Yahoo start Date is 19700101
    year_start = int(start_date[0:4])
    month_start = int(start_date[4:6])
    day_start = int(start_date[-2:])
    t1 = dt.datetime(year_start, month_start, day_start, 0, 0)
    period_1 = str(int((t1-dt.datetime(1970,1,1)).total_seconds()))
    
    year_end = int(end_date[0:4])
    month_end = int(end_date[4:6])
    day_end = int(end_date[-2:])
    t2 = dt.datetime(year_end, month_end, day_end, 0, 0)
    period_2 = str(int((t2-dt.datetime(1970,1,1)).total_seconds()))
    print(period_1,period_2)
    
    # Build URL
    url = 'https://finance.yahoo.com/quote/' + stock + '/history?period1='+ period_1 +'&period2='+ period_2 + '&interval=1d&filter=history&frequency=1d'
    return url
    print(url)
    
#url = get_url()

#%%
start='1/1/2018'
end='1/01/2019'

#%%

symbols = ['IWD', 'IWC', 'SPY', 'DEM']#, 'CLY'] This latest one will be added later
#symbols = ['BA', 'C', 'AAL', 'NFLX']
prices = pd.DataFrame(index=pd.date_range(start, end))

#%%


dict_of_df = {}

for symbol in symbols:
    #portfolio = web.DataReader(name=symbol, data_source='quandl', start=start, end=end)
    url = get_url(symbol,'20180101','20190101')
    # Remember the parser
    data = r.get(url)
    page_data = soup(data.text, 'html.parser')
    table = page_data.findAll('table')[0]
    df = pd.read_html(str(table))
    my_table = df[0]
    # Rmove rows with the Dividend atribution
    my_table = my_table[~my_table['Open'].str.contains("Dividend")]
    my_table = my_table.iloc[:-1,:]
    my_table.reset_index(drop = True, inplace=True)
    my_table.iloc[:,1:] = my_table.iloc[:,1:].astype('float64')
    my_table['Adj Close**'] = my_table['Adj Close**'].apply(pd.to_numeric)
    #my_table
    key_name = 'df_new_'+str(symbol)
    
    dict_of_df[key_name] = my_table
    #close = my_table[['Adj Close**']]
    #close = close.rename(columns={'Adj Close**': symbol})
    #prices = prices.join(close)
    #portfolio.to_csv("~/workspace/{}.csv".format(symbol))

#%%
# Create ab empty dataframe to store the prices

#prices = dict_of_df


#prices = dict_of_df[teste]['Adj Close**']
prices = pd.DataFrame(dict_of_df[list(dict_of_df.keys())[0]]['Date'])

for j,i in enumerate(list(dict_of_df.keys())):
    print(j, i)
    teste = dict_of_df[i][['Date','Adj Close**']]
    #dada = teste.rename(columns = {"Date":"Date","Adj Close**":list(dict_of_df.keys())[j]})
    dada = teste.rename(columns = {"Date":"Date","Adj Close**":re.sub('^df_new_', '', list(dict_of_df.keys())[j])})
    #prices.merge(teste, left_on = "Date", right_index = False, how='left')#, on='mukey', how='left')
    prices = pd.concat([prices, dada],axis = 1 ,join = 'inner')
    prices = prices.loc[:, ~prices.columns.duplicated()]
    #prices = prices.join(teste)
    print(j, i)

    
#%%
prices = prices.dropna()
prices.drop("Date", axis=1, inplace=True)
risk_v = get_risk(prices)
return_v = get_return(prices)
fig, ax = plt.subplots()
ax.scatter(x=risk_v, y=return_v, alpha=0.5)
ax.set(title='Return and Risk', xlabel='Risk', ylabel='Return')
for i, symbol in enumerate(symbols):
    ax.annotate(symbol, (risk_v[i], return_v[i]))
plt.show()

#%%

# Build portfolio with random weights

def random_weights(n):
    weights = np.random.rand(n)
    return weights / sum(weights)

def get_portfolio_risk(weights, normalized_prices):
    portfolio_val = (normalized_prices * weights).sum(axis=1)
    portfolio = pd.DataFrame(index=normalized_prices.index, data={'portfolio': portfolio_val})
    return (portfolio / portfolio.shift(1) - 1).dropna().std().values[0]

def get_portfolio_return(weights, normalized_prices):
    portfolio_val = (normalized_prices * weights).sum(axis=1)
    portfolio = pd.DataFrame(index=normalized_prices.index, data={'portfolio': portfolio_val})
    ret = get_return(portfolio)
    return ret[0]

#%%

risk_all = np.array([])
return_all = np.array([])

#%%

#Not deeply tested - JA

# for demo purpose, plot 3000 random portoflio
np.random.seed(0)
normalized_prices = prices / prices.iloc[0, :]

#%%

# 3000 options of weights
for _ in range(0, 3000):
    weights = random_weights(len(symbols))
    portfolio_val = (normalized_prices * weights).sum(axis=1)
    portfolio = pd.DataFrame(index=prices.index, data={'portfolio': portfolio_val})
    risk = get_risk(portfolio)
    ret = get_return(portfolio)
    risk_all = np.append(risk_all, risk)
    return_all = np.append(return_all, ret)
    p = get_portfolio_risk(weights=weights, normalized_prices=normalized_prices)
fig, ax = plt.subplots()
ax.scatter(x=risk_all, y=return_all, alpha=0.5)
ax.set(title='Return and Risk', xlabel='Risk', ylabel='Return')



for i, symbol in enumerate(symbols):
    ax.annotate(symbol, (risk_v[i], return_v[i]))
ax.scatter(x=risk_v, y=return_v, alpha=0.5, color='red')
ax.set(title='Return and Risk', xlabel='Risk', ylabel='Return')
ax.grid()
plt.show()

#%%

# Efficient Frontier

# Optimize the weights

# optimizer
def optimize(prices, symbols, target_return=0.1):
    normalized_prices = prices / prices.ix[0, :]
    # all Symbols have the same opportunity
    init_guess = np.ones(len(symbols)) * (1.0 / len(symbols))
    #Between 0 and 1
    bounds = ((0.0, 1.0),) * len(symbols)
    weights = minimize(get_portfolio_risk, init_guess,
                       args=(normalized_prices,), method='SLSQP',
                       options={'disp': False},
                       constraints=({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)},
                                    {'type': 'eq', 'args': (normalized_prices,),
                                     'fun': lambda inputs, normalized_prices:
                                     target_return - get_portfolio_return(weights=inputs,
                                                                          normalized_prices=normalized_prices)}),
                       bounds=bounds)
    return weights.x


optimal_risk_all = np.array([])
optimal_return_all = np.array([])


for target_return in np.arange(0.005, .0402, .0005):
    opt_w = optimize(prices=prices, symbols=symbols, target_return=target_return)
    optimal_risk_all = np.append(optimal_risk_all, get_portfolio_risk(opt_w, normalized_prices))
    optimal_return_all = np.append(optimal_return_all, get_portfolio_return(opt_w, normalized_prices))
# plotting
fig, ax = plt.subplots()
# random portfolio risk return
ax.scatter(x=risk_all, y=return_all, alpha=0.5)
# optimal portfolio risk return
for i, symbol in enumerate(symbols):
    ax.annotate(symbol, (risk_v[i], return_v[i]))
ax.plot(optimal_risk_all, optimal_return_all, '-', color='green')
# symbol risk return
for i, symbol in enumerate(symbols):
    ax.annotate(symbol, (risk_v[i], return_v[i]))
ax.scatter(x=risk_v, y=return_v, color='red')
ax.set(title='Efficient Frontier', xlabel='Risk', ylabel='Return')
ax.grid()
plt.savefig('return_risk_efficient_frontier.png', bbox_inches='tight')



