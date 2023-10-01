# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 17:58:58 2023

@author: praveen
"""
## Ref - https://www.youtube.com/watch?v=fk38oX9GsxE&list=PLqpCwow11-OqqfELduCMcRI6wcnoM3GAZ&index=10

import numpy as np

S0 = 100 # Initial Stock Price
K = 100 # Strike Price
T = 1 # time to Maturity in years
B = 125 # Up and Out Barrier
RFR = 0.03 # Risk free rate
Vol = 0.2 # Volatility
N = 100 # no.of.time steps
M =1000 #no.of simulations

#Constants
dt = T/N
nudt = (RFR-0.5*Vol**2)*dt
volsdt = Vol*np.sqrt(dt)
erdt = np.exp(RFR*dt)

sum_CT = 0
sum_CT2= 0

# Monte Carlo Method

for i in range(M):
    BARRIER = False
    St = S0
    
    for j in range(N):
        epsilon = np.random.normal()
        Stn = St*np.exp(nudt + volsdt*epsilon)
        St = Stn
        
        if St >= B:
            BARRIER = True
            break
        
    if BARRIER:
        CT = 0
        
    else:
        CT = max(0,K-St)
        
    sum_CT = sum_CT + CT
    sum_CT2 = sum_CT2 + CT*CT
    

C0 = np.exp(-RFR*T)*sum_CT/M
sigma = np.sqrt((sum_CT2 - sum_CT * sum_CT/M)*np.exp(-2*RFR*T)/(M-1))
SE = sigma/np.sqrt(M)

print('Value of Option is {}'.format(np.round(C0,2)))
print(SE)