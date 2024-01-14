# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:02:21 2023

@author: praveen
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 18:33:53 2023

@author: praveen
"""

from abc import ABCMeta, abstractmethod
import numpy as np

class Finitediff(metaclass = ABCMeta):
    """
    Parameters
    ----------
    S0 : Enter Current Stock Price.
    K : Enter Strike Price.
    TTM : Enter Time to Maturity of the Option in days.    
    Opttype : Enter if it's Call or Put Option.
    Vol : Enter Annualalized Volatility for Pricing.       
    RFR : Enter Annualized Risk Free Rate. 
    DS = Irrespective of this input code places deltaS at 0.1% of previous S
    DT = Enter DT for time step: max 1day or lower. Lower values should be divisors of 1
    -------


    """
    def __init__(self,S0,K,TTM,Opttype, Vol, RFR, DS, DT):
        
        self.S0 = S0
        self.K = K
        self.TTM = TTM
        self.Opttype = Opttype
        self.Vol = Vol
        self.RFR = RFR
        self.DS = DS
        self.DT = DT
        
    def validateinputs(self):
        opttypeinputs = ["C","P","Call","Put"]
        if self.Opttype not in opttypeinputs:
            print("Enter Option type as \"C\" or \"P\" or \"Call\" or\"Put\"")
        if type(self.S0)!= int and type(self.S0)!= float:
            print("Enter a numeric stock price value")
        if type(self.K)!= int and type(self.K)!= float:
            print("Enter a numeric strike price value")
             
    @abstractmethod
    def creategrid(self):
        raise NotImplementedError('Implementation Required!')
        
    @abstractmethod
    def setboundary(self):
        raise NotImplementedError('Implementation Required!')
      
    @abstractmethod
    def calculatecoeff(self):
        raise NotImplementedError('Implementation Required!')
        
    @abstractmethod
    def traversegrid(self):
        raise NotImplementedError('Implementation Required!')
        
    @abstractmethod
    def pricev(self):
        raise NotImplementedError('Implementation Required!')
         
        
            
class FDiffeuoptexp(Finitediff):
    
    def creategrid(self):
        deltaS = self.DS * self.S0
        self.sstep = 0.5/self.DS
        sstep = 0.5/self.DS
        self.Srange = np.zeros(31)
        j = 15
        for i in range(31):
            if i<15:
                self.Srange[i]=(self.S0)/(1.1**j)
                j=j-1
            elif i == 15:
                self.Srange[i] = self.S0
                j=j+1
            elif i>15:
                self.Srange[i]=(self.S0)*(1.1**j)
                j=j+1

        self.tstep = self.TTM/self.DT
        tstep = self.TTM/self.DT
        self.fdiffgrid = np.zeros ((31,int(tstep)+1),dtype=float)
        return self.fdiffgrid
    
        
    def setboundary(self):
        if self.Opttype == 'C' or self.Opttype == 'Call':
            for i in range(31):
                self.fdiffgrid[i,int(self.tstep)] = max(self.Srange[i] - self.K,0)
        elif self.Opttype == 'P' or self.Opttype == 'Put':
            for i in range(31):
                self.fdiffgrid[i,int(self.tstep)] = max(self.K - self.Srange[i],0)
        else:
              print('Need Option type to set up boundary conditions')
             
        return self.fdiffgrid
        
                
    def calculatecoeff(self):
        yearDT = float(self.DT/365)
        #0.09531 is change in Ln S in the grid.
        x = ((self.Vol**2)*yearDT)/(2*(0.09531**2))
        self.a = x + (self.RFR - (0.5*(self.Vol**2)))*(yearDT/(2*0.09531))
        self.b = 1- ((yearDT*(self.Vol**2))/(0.09531**2))
        self.c = x - (self.RFR - (0.5*(self.Vol**2)))*(yearDT/(2*0.09531))
        

            
    def traversegrid(self):
        yearDT = float(self.DT/365)
        discf = 1/(1+ (self.RFR*yearDT))
        sstep = 0.5/self.DS
        tstep = self.TTM/self.DT
        deltaSCall = self.Srange[30]-self.Srange[29]
        deltaSPut = self.Srange[1]-self.Srange[0]
        if self.Opttype =='C' or 'Call':
            
            for i in range(int(tstep)):

                for j in range(31):
                    
                    if j == 0:
                        self.fdiffgrid[j,int(tstep)-i-1] = self.c*0+\
                            self.b*self.fdiffgrid[j,int(tstep)-i]+\
                                self.a*self.fdiffgrid[j+1,int(tstep)-i]
                    elif j == 30:
                        self.fdiffgrid[j,int(tstep)-i-1] = self.c*self.fdiffgrid[j-1,int(tstep)-i]+\
                            self.b*self.fdiffgrid[j,int(tstep)-i]+\
                                self.a*(self.fdiffgrid[j,int(tstep)-i] + deltaSCall)
                    else:
                        self.fdiffgrid[j,int(tstep)-i-1] = self.c*self.fdiffgrid[j-1,int(tstep)-i]+\
                            self.b*self.fdiffgrid[j,int(tstep)-i]+\
                                self.a*self.fdiffgrid[j+1,int(tstep)-i]
                            
        elif self.Opttype == 'P' or 'Put':
            
            for i in range(int(tstep)):

                for j in range(31):
                    
                    if j == 0:
                        self.fdiffgrid[j,int(tstep)-i-1] = self.c*(self.fdiffgrid[j,int(tstep)-i]+deltaSPut)+\
                            self.b*self.fdiffgrid[j,int(tstep)-i]+\
                                self.a*self.fdiffgrid[j+1,int(tstep)-i]
                    elif j == 30:
                        self.fdiffgrid[j,int(tstep)-i-1] = self.c*self.fdiffgrid[j-1,int(tstep)-i]+\
                            self.b*self.fdiffgrid[j,int(tstep)-i]+\
                                self.a[j]*0
                    else:
                        self.fdiffgrid[j,int(tstep)-i-1] = self.c*self.fdiffgrid[j-1,int(tstep)-i]+\
                            self.b*self.fdiffgrid[j,int(tstep)-i]+\
                                self.a*self.fdiffgrid[j+1,int(tstep)-i]
                
            return self.fdiffgrid
        
    def pricev(self,x):
        x.creategrid()
        x.setboundary()
        x.calculatecoeff()
        x.traversegrid()
        self.pricev = self.fdiffgrid[15,0]
        return self.pricev


        

sample = FDiffeuoptexp(23, 22, 30, 'P', 0.4, 0.03, 0.02, 2)
sample.pricev(sample)
