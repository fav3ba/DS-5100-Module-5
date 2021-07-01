# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:56:25 2021

@author: fav21
"""

# load modules
import numpy as np
import pandas as pd

# risk-free Treasury rate
R_f = 0.0175 / 252

# read in the market data
data = pd.read_csv('capm_market_data.csv')

data.head()

df = data.drop(columns=['date'])

daily = df.pct_change(axis=0)
daily.dropna(inplace=True)

daily.head(5)

spy = daily.spy_adj_close.values
print("SPY first five:", spy[0:5])
aapl = daily.aapl_adj_close.values
print("AAPL first five:", aapl[0:5])

ex_spy = spy-R_f
ex_aapl = aapl-R_f

print("SPY excess:",ex_spy[-5:])
print("AAPL excess:",ex_aapl[-5:])

import matplotlib.pyplot as plt

plt.scatter(x=ex_spy,y=ex_aapl)

x = ex_spy.reshape(-1,1)
y = ex_aapl.reshape(-1,1)

beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(),x)),x.transpose()),y)
beta = beta[0][0]
print(beta)

def beta_sensitivity(x_val,y_val):
    estimates = []
    for i in range(0,len(x_val)):
        x_temp = np.delete(x_val,i).reshape(-1,1)
        y_temp = np.delete(y_val,i).reshape(-1,1)
        beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_temp.transpose(),x_temp)),x_temp.transpose()),y_temp)
        beta = beta[0][0]
        estimate = (i,beta)
        estimates.append(estimate)
    return(estimates)

betas = beta_sensitivity(ex_spy,ex_aapl)
betas[:5]