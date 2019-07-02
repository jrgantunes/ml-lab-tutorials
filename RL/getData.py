# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:43:01 2019

@author: Jorge Antunes


Yahoo Finance
The goal of this file is to import the data from Yahoo Finance
"""

#%%
#Change Path
import os
os.chdir('D:\OneDrive - NOVAIMS\PhD\Codigo\ArticleRRL')

#%%

# import the libraries
import requests as r
from bs4 import BeautifulSoup as soup
import pandas as pd
import datetime as dt
import numpy as np
import math

#%%

# Initial parameters
# at this stage we start to introduce the concepts from Almahdi and Yang 2017

n_returns = 5

# number of shares
niu = 100
# weight
theta_1 = 0.5


# To develop later
x_t_global = 50
Theta_global = 6

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

# dates yyyymmdd
# the end date is exclusive
url = get_url('IWD','20180101','20190101')

#%%
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
#my_table

#%%
#Remove the last row *Close price adjusted for splits.**Adjusted cl...
#my_table = my_table.iloc[:-1,:]

my_table.iloc[:,1:] = my_table.iloc[:,1:].astype('float64')
my_table['Close*'] = my_table['Close*'].apply(pd.to_numeric)
#%%

# Log Return

my_table['log_price'] = np.log(my_table['Close*'])

log_return =[]

for i in range(len(my_table)-1):
    log_return.append(my_table.loc[i+1,'log_price'] - my_table.loc[i,'log_price'])
    
log_return.insert(0,0)
my_table['log_return'] = log_return
del log_return

#%%

# Return at time t Eq. 3
# Assume no chnage 
my_table['Return'] = niu * my_table['log_return']
my_table['sqrt_return'] = my_table['Return']**2
# Calculate Reward and Reward Squared

#teste =[]
#
#for i in range(len(my_table)-1):
#    teste.append(my_table.loc[i+1,'Close*'] - my_table.loc[i,'Close*'])
#    
#teste.insert(0,0)
#my_table['Reward'] = teste
#my_table['SQRTRew'] = my_table['Reward']*my_table['Reward']
#del teste

#%%

# log price
#my_table['log_price'] = np.log(my_table['Close*'])

#teste =[]
#
#for i in range(len(my_table)-1):
#    teste.append(my_table.loc[i+1,'log_price'] - my_table.loc[i,'log_price'])
#    
#teste.insert(0,0)
#my_table['Reward'] = teste
#my_table['SQRTRew'] = my_table['Reward']*my_table['Reward']
#del teste


#%%

# Define A & B

my_table['A'] = 0
my_table['B'] = 0

a_list = [0.0001,0.0001,0.0001,0.0001]
b_list = [0.0001,0.0001,0.0001,0.0001]

for i in range(n_returns-1,len(my_table)):
    a_list.append(sum(my_table.loc[i-(n_returns-1):i,'Return']) / n_returns)
    b_list.append(sum(my_table.loc[i-(n_returns-1):i,'sqrt_return']) / n_returns)

my_table['A'] = a_list
my_table['B'] = b_list

#%%

# Define the first derivative 

#my_table['dF_dTheta'] = 0



#%%

# Log sigmoid function Eq. 7

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


#%%
# number of shares
niu = 100
# weight
theta_1 = 0.5
theta_f = 0.5
#markets
theta_m1 = 0.5

theta_list = [theta_m1, theta_f, theta_1]
theta_array = np.array(theta_list)

#Dummy_values
profit_1 = 5
Ft_1 = 1
dft_1_dtheta = 3
constant = 1

x_list = [profit_1,Ft_1, constant]
x_array = np.array(x_list)


# Start the dF_dTheta
my_table['dF_dTheta'] = 0
my_table['theta_1'] = 0.5
my_table['theta_f'] = 0.5
my_table['theta_m1'] = 0.5
my_table['dCT'] = 0.0
my_table['Ft'] = 1
# Long Portfolio
# dRt / dFt = 0
# dFt / dtheta  not necessary


left_side = (1- sigmoid(np.dot(x_array,theta_array))**2)
right_side = (x_array + theta_array * dft_1_dtheta)
dF_theta = left_side * right_side

# dRt / dFt-1 = -mu * rt
delta = 0
#dRt_Ft_1 = -niu * 

#%%
my_table['DCT_dAn'] = 0 

dF_theta = []


#for i in range(n_returns-1,len(my_table)):
for i in range(n_returns-1,len(my_table)):
    # i starts at zero but time at 1
    for j in range(n_returns):
        temp_table = pd.DataFrame(index=range(n_returns),columns=['dCTA','dCTB','dRtFt'], dtype='float')
        if my_table.loc[i,'A'] < 0:
            #my_table.loc[i,'DCT_dAn'] =  - 1 * (2 * j * my_table.loc[i,'B'] * my_table.loc[i,'A']) / (((j-1)*(my_table.loc[i,'A']**2)+my_table.loc[i,'B'])**2)
            # the is to make the time and the datframe order to match that's why I sum and discount
            temp_table.loc[j,'dCTA'] =  - 1 * (2 * (j+1) * my_table.loc[i-(n_returns-j),'B'] * my_table.loc[i-(n_returns-j),'A']) / (((j+1-1)*(my_table.loc[i-(n_returns-j),'A']**2)+my_table.loc[i-(n_returns-j),'B'])**2)
            temp_table.loc[j,'dCTB'] = ((j+1)*my_table.loc[i-(n_returns-j),'A']**2)/((my_table.loc[i-(n_returns-j),'B'] + (j+1-1)*(my_table.loc[i-(n_returns-j),'A']**2))**2)
        else:
            temp_table.loc[j,'dCTA'] = (200000 * (j+1) * my_table.loc[i-(n_returns-j),'B'] * my_table.loc[i-(n_returns-j),'A'] * (100000 * math.log(my_table.loc[i-(n_returns-j),'A'] / ((my_table.loc[i-(n_returns-j),'B'] - my_table.loc[i-(n_returns-j),'A']**2)**(1/2)))+ 50000 * math.log(j+1) + 13519) )/ (((my_table.loc[i-(n_returns-j),'A']**2 - my_table.loc[i-(n_returns-j),'B'])**2) * (100000 * math.log(my_table.loc[i-(n_returns-j),'A'] / ((my_table.loc[i-(n_returns-j),'B'] - my_table.loc[i-(n_returns-j),'A']**2)**(1/2)))+ 50000 * math.log(j+1) + 63519)**2)
            temp_table.loc[j,'dCTB'] = (100000 * (j+1) * my_table.loc[i-(n_returns-j),'A']**2 * (100000 * math.log(my_table.loc[i-(n_returns-j),'A'] / ((my_table.loc[i-(n_returns-j),'B'] - my_table.loc[i-(n_returns-j),'A']**2)**(1/2)))+ 50000 * math.log(j+1) + 13519)) / (((my_table.loc[i-(n_returns-j),'B']- my_table.loc[i-(n_returns-j),'A']**2 )**2) * (100000 * math.log(my_table.loc[i-(n_returns-j),'A'] / ((my_table.loc[i-(n_returns-j),'B'] - my_table.loc[i-(n_returns-j),'A']**2)**(1/2)))+ 50000 * math.log(j+1) + 63519)**2)
        try:
            if my_table.loc[i-(n_returns-j),'Ft'] != my_table.loc[i-(n_returns-j) - 1,'Ft']:
                temp_table.loc[j,'dRtFt'] = -niu * delta
            else:
                temp_table.loc[j,'dRtFt'] = 0
        except:
            temp_table.loc[j,'dRtFt'] = 0
        
        # Update thetas
        try:
            x_list = [my_table.loc[i-(n_returns-j),'log_return'], my_table.loc[i-(n_returns-j)-1,'Ft'], 1]
            x_array = np.array(x_list)
        except:
            x_array = np.array([0.000000001,1,1])
        
        try:   
            theta_list = [my_table.loc[i-(n_returns-j),'theta_m1'], my_table.loc[i-(n_returns-j)-1,'theta_f'], my_table.loc[i-(n_returns-j)-1,'theta_1']]
            theta_array = np.array(theta_list)
        except:
            theta_array = np.array([1,1,1])
        
        left_side = (1- sigmoid(np.dot(x_array,theta_array))**2)
        right_side = (x_array + theta_array * dft_1_dtheta)
        dF_theta.append(left_side * right_side)
        

    
         


