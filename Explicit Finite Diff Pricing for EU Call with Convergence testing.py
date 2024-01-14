# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 20:42:08 2024

@author: praveen
"""
import numpy as np

def fdiffeucall(S,K,M,r,div,sig,N,Nj,dx):
    """
    
    S: Initial Stock price\t
    K: strike price\t
    M: Maturity in years\t
    r: annualized rate of interest\t
    div: annualized dividend yield\t
    sig: annualized standard deviation\t
    N: no.of time steps\t
    Nj: no.of.up/down moves.Grid will have 2*Nj + 1 stock price values at each time step\t
    dx: step size in %age for stock price move in the grid\t
    Output: Call Option Price


    """
    
    dt = M/N
    edx = np.exp(dx)
    nu = r-div-(0.5*(sig**2))
    pu = 0.5*dt*(((sig/dx)**2) + (nu/dx))
    pd = 0.5*dt*(((sig/dx)**2) - (nu/dx))
    pm = 1.0-dt*((sig/dx)**2) - (r*dt)
    
    # Generate asset prices and call moneyness at maturity
    N = int(N)
    Nj = int(Nj)
    st = np.zeros((2*Nj+1,1))
    eucall = np.zeros((2*Nj+1,N+1))
    st[2*Nj] = S*np.exp(-Nj*dx)
    eucall[2*Nj,N] = max(st[2*Nj] - K,0)
    for i in range(2*Nj-1,-1,-1):
        st[i] = st[i+1] * edx
        eucall[i,N] = max(st[i] - K,0)

    #populatting the grid backwards
    for i in range(N-1,-1,-1):
        for j in range(2*Nj-1,0,-1):
            eucall[j,i] = pu*eucall[j-1,i+1] + pm*eucall[j,i+1] + pd*eucall[j+1,i+1]
            
            
        #boundary conditions
            eucall[2*Nj,i] = eucall[2*Nj-1,i]
            eucall[0,i] = eucall[1,i] + st[0] - st[1]
    return eucall[Nj,0]
            
            

#Blackscholes value

import math
import numpy as np
from scipy.stats import norm
import pandas as pd
print(date.today())

def BlackScholes(S0, K, TTM, Vol, IR, Opttype):
    d1 = (1/Vol*np.sqrt(TTM))*(np.log(S0/K) + (IR+ 0.5*Vol**2)*TTM)
    d2 = d1- (Vol*np.sqrt(TTM))
    if Opttype == 'C':
        price = (S0*norm.cdf(d1)) - ((np.exp(-IR*TTM)*K) * (norm.cdf(d2)))
    elif Opttype =='P':
        price = ((np.exp(-IR*TTM)*K) * (norm.cdf(-d2)))-(S0*norm.cdf(-d1))
    else:
        price = print("Enter a valid option 'C' for Call and 'P' for Put")
        
    return price


#Checking for Convergence to BS Value at various levels of dx with varying time steps

#Setting Inputs
Inputs = np.zeros((100,9))
Inputs[:,0:9] = [100,100,1,0.06,0,0.2,10,50,0.001]
Output = np.zeros((100,7))
Output[:,0] = BlackScholes(100, 100,1, 0.2, 0.06, 'C')
Inputs[i,8] = 0.01
for i in range(1,100):
    Inputs[i,8] = Inputs[i-1,8] + 0.001

timesteps = [2,3,5,10,25,50]

#Running Ouput for different Inputs
for j in range(6):   
    Inputs[:,7] = timesteps[j]
    print(timesteps[j])
    for i in range(100):
        Output[i,j+1] = fdiffeucall(Inputs[i,0],Inputs[i,1],Inputs[i,2],Inputs[i,3],Inputs[i,4],\
                                  Inputs[i,5],Inputs[i,6],Inputs[i,7],Inputs[i,8])
                    

#Plotting Convergence at various levels of dx
#Choice of timesteps has less impact on final price compared to choice of dx

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.figure(figsize=(10,6))
plt.xlabel('Choice of dx')
plt.ylabel('Option Price')
plt.title('Convergence to Black Scholes Price at different choice of dx')
plt.ylim(-50,15)
plt.plot(Inputs[:,8],Output[:,0],lw=1.5,label ='Black Scholes Option Price')
plt.plot(Inputs[:,8],Output[:,1],lw=1.5,label ='finite diff price timesteps = 2')
plt.plot(Inputs[:,8],Output[:,2],lw=1.5,label ='finite diff price timesteps = 3')
plt.plot(Inputs[:,8],Output[:,3],lw=1.5,label ='finite diff price timesteps = 5')
plt.plot(Inputs[:,8],Output[:,4],lw=1.5,label ='finite diff price timesteps = 10')
plt.plot(Inputs[:,8],Output[:,5],lw=1.5,label ='finite diff price timesteps = 25')
plt.plot(Inputs[:,8],Output[:,6],lw=1.5,label ='finite diff price timesteps = 50')
plt.grid(False)
plt.legend(loc=0)
