# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:10:11 2024

@author: praveen
"""

import numpy as np

S = 100
K = 100
M = 1
r = 0.06
div = 0
N = 20
Nj = 10
dx = 0.05
sig = 0.2


def solvetridiagonalmatrix(C,pu,pm,pd,lambda_L,lambda_U):
    j = len(C) - 1
    pmp = np.zeros((j,1))
    pp = np.zeros((j,1))
    pmp[j-1,0] = pm + pd
    pp[j-1,0] = C[j-1,1] + pd*lambda_L
    
    for i in range (j-2, 0,-1):
        pmp[i,0] = pm-((pu/pmp[i+1,0])*pd)
        pp[i,0] = C[i,1] - (pp[i+1,0] * (pd/pmp[i+1,0]))
        
    C[0,0] = (pp[1,0] + pmp[1,0] * lambda_U) / (pu + pmp[1,0])
    C[1,0] = C[0,0] - lambda_U
    
    for i in range(2,len(C)-1,1):
        C[i,0] = (pp[i,0] - pu*C[i-1,0])/pmp[i,0]
        
    C[j,0] = C[j-1,0] - lambda_L
    
        




def fdiffAPUT(S,K,M,r,div,sig,N,Nj,dx):
    
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
    Output: Call Option Price\t


    """
    
    dt = M/N
    edx = np.exp(dx)
    nu = r-div-(0.5*(sig**2))
    pu = -0.5*dt*(((sig/dx)**2) + (nu/dx))
    pd = -0.5*dt*(((sig/dx)**2) - (nu/dx))
    pm = 1.0 + dt*((sig/dx)**2) + (r*dt)
    
    # Generate asset prices and Put moneyness at maturity
    N = int(N)
    Nj = int(Nj)
    st = np.zeros((2*Nj+1,1))
    APut = np.zeros((2*Nj+1,N+1))
    st[2*Nj] = S*np.exp(-Nj*dx)
    APut[2*Nj,N] = max(K-st[2*Nj],0)
    for i in range(2*Nj-1,-1,-1):
        st[i] = st[i+1] * edx
        APut[i,N] = max(K-st[i,0],0)
        
    #Setting Boundary Conditions
    
    lambda_L = -1* (st[2*Nj - 1] - st[2*Nj]) 
    lambda_U = 0
        
    #populatting the grid backwards
    for i in range(N-1,-1,-1):
        solvetridiagonalmatrix(APut[:,i:i+2],pu,pm,pd,lambda_L,lambda_U)
        
    # Populating early exercise payoff
        
        for j in range(2*Nj,-1,-1):
            APut[j,i] = max(APut[j,i], K - st[j])
        
    return APut[Nj,0]


#Binomial Option Pricing

class Binomoptpricing:
    
    """
     
     :param S0: Initial Stock Price
     :param K: Strike Price of the Option
     :param Maturity: Maturity in Years
     :param Vol: Annualized Vol for pricing
     :param IR: Annualized risk free rate
     :param Opttype: 'C' or'Call' for Call and 'P' or 'Put' for Put
     :param Etype: 'A' for American or ''
     :param Tstep: DESCRIPTION
     :return: DESCRIPTION

    """
    
    def __init__(self,S0, K, Maturity, Vol, IR, Opttype, Etype, Tstep):
   

        self.S0 = S0
        self.K = K
        self.M = Maturity
        self.Vol = Vol
        self.IR = IR
        self.Opttype = Opttype
        self.Etype = Etype
        self.Tstep = Tstep
        # return self.stockpricetree()
    
        
    def validateinputs(self):
        opttypeinputs = ["C","P","Call","Put"]
        Etypeinputs = ["A","E"]
        
        if self.Opttype not in opttypeinputs:
            print("Enter Option type as \"C\" or \"P\" or \"Call\" or\"Put\"")
        if self.Etype not in Etypeinputs:
            print("Enter Etype as \"A\" for American Option or \"E\" for European Option")
        else:
            print("Inputs are fine")
        
    def stockpricetree(self):
        import numpy as np
        import math as math
        h= round(self.M/self.Tstep,15)
        sigma = self.Vol*np.sqrt(h)
        u = np.exp(sigma)
        d = 1/u
        RFR = np.exp(self.IR*h)
        DF = 1/RFR
        q = (RFR-d)/(u-d)
        Stocktree = np.zeros((self.Tstep+1, self.Tstep+2),dtype = float)
        Moneyness = np.zeros((self.Tstep+1, self.Tstep+2),dtype = float)
        Optvalue = np.zeros((self.Tstep+1, self.Tstep+2),dtype = float)
        Stocktree[0,0] = self.S0
        Call = ["C","Call"]
        Put = ["P","Put"]
        for i in range (1, self.Tstep + 1):
            z = i + 1
            for j in range (1, z+1):
                Stocktree[i,j] = self.S0 *(u**(i+1-j))*(d**(i-(i+1-j)))
                if self.Opttype in Call:
                    Moneyness[i,j] = max(Stocktree[i,j]-self.K,0)
                elif self.Opttype in Put:
                    Moneyness[i,j] = max(self.K-Stocktree[i,j],0)
        self.Stocktree = Stocktree
        self.Moneyness = Moneyness
                    
        if self.Etype == "E":
            for i in reversed(range (1, self.Tstep)):
                z = i + 1
                for j in range (1, z+1):
                    if i == self.Tstep-1:
                        Optvalue[i,j] = (self.Moneyness[i + 1,j]*q*DF) + \
                            (self.Moneyness[i+1,j+1]*(1-q)*DF)
                    else:
                        Optvalue[i,j] = (Optvalue[i + 1,j]*q*DF) + \
                            (Optvalue[i+1,j+1]*(1-q)*DF)
                            
            Optvalue[0,0] = (Optvalue[1,1]*q*DF) + \
                (Optvalue[1,2]*(1-q)*DF)
            self.Optvalue = Optvalue
            print(Optvalue[0,0])
        
        elif self.Etype == "A":                      
            for i in reversed(range (1, self.Tstep)):
                z = i + 1
                for j in range (1, z+1):
                    if i == self.Tstep-1:
                        Optvalue[i,j] = max((self.Moneyness[i + 1,j]*q*DF) + \
                            (self.Moneyness[i+1,j+1]*(1-q)*DF),self.Moneyness[i,j])
                    else:
                        Optvalue[i,j] = max((Optvalue[i + 1,j]*q*DF) + \
                            (Optvalue[i+1,j+1]*(1-q)*DF),self.Moneyness[i,j])
                            
            Optvalue[0,0] = (Optvalue[1,1]*q*DF) + \
                (Optvalue[1,2]*(1-q)*DF)
            self.Optvalue = Optvalue
            return Optvalue[0,0]








#Sample Ouputs
print("Sample Outputs from Binomial and Finite Difference Methods are")
binom = Binomoptpricing(S,K,M,sig,r,"Put","A",100)
binomoutput = binom.stockpricetree()
print(binomoutput)
print(fdiffAPUT(S,K,M,r,div,sig,N,Nj,dx))




#Convergence and Comparison to Binomial Tree

#Setting Inputs

Inputs = np.zeros((100,9))
Inputs[:,0:9] = [100,100,1,0.06,0,0.2,10,50,0.001]
Output = np.zeros((100,7))
Output[:,0] = binom.stockpricetree()
Inputs[0,8] = 0.01
for i in range(1,100):
    Inputs[i,8] = Inputs[i-1,8] + 0.001

timesteps = [2,3,5,10,25,50]

#Running Ouput for different Inputs
for j in range(6):   
    Inputs[:,7] = timesteps[j]
    print(timesteps[j])
    for i in range(100):
        Output[i,j+1] = fdiffAPUT(Inputs[i,0],Inputs[i,1],Inputs[i,2],Inputs[i,3],Inputs[i,4],\
                                  Inputs[i,5],Inputs[i,6],Inputs[i,7],Inputs[i,8])
                    

#Plotting Convergence at various levels of dx
#Choice of timesteps has less impact on final price compared to choice of dx

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.xlabel('Choice of dx')
plt.ylabel('Option Price')
plt.title('Convergence Of Price at different choice of dx')
plt.ylim(-10,100)
plt.plot(Inputs[:,8],Output[:,0],lw=1.5,label ='Binomial Price with 100 time steps')
plt.plot(Inputs[:,8],Output[:,1],lw=1.5,label ='finite diff price timesteps = 2')
plt.plot(Inputs[:,8],Output[:,2],lw=1.5,label ='finite diff price timesteps = 3')
plt.plot(Inputs[:,8],Output[:,3],lw=1.5,label ='finite diff price timesteps = 5')
plt.plot(Inputs[:,8],Output[:,4],lw=1.5,label ='finite diff price timesteps = 10')
plt.plot(Inputs[:,8],Output[:,5],lw=1.5,label ='finite diff price timesteps = 25')
plt.plot(Inputs[:,8],Output[:,6],lw=1.5,label ='finite diff price timesteps = 50')
plt.grid(False)
plt.legend(loc=0)
