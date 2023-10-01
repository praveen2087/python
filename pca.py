# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#DF4 has the eigenvalues.
#DF5 has the eigen vector.Row 1 pertains to 0.5yr Row 2 pertains to 1 yr and so on.
#the PCA numpy array has time series of Principal components. Full row for 1 timepoint.
# Multiplying PC with corresponding Eigen Vector should show the change in rates for the time point/
# For example eigen vector pertaining to 0.5y can be multiplied with Principal components in timepoint 1 to derive change in /
#0.5y rates at timepoint 1
#Using 1st 3 or 4 eigen vectors/ PCs should be sufficient as they explain majority of the variation


#How can PCA be used. Can be used by specifying shocks on Principal components and by using the eigen vectors./
# It can be translated into an YC scenario.

#The eigenvectors/ PCs can be built either from the correlation matrix or from the co-variance matrix.Both work
#the results from both need not align ie, same Eigen vectors/PCs

#File with input is uploaded in GitHub ie,Path variable

import pandas as pd
import numpy as np
from numpy.linalg import eig
path = "C:\\Praveen\\Programming\\chapter2\\II.2\\Case Study II.2_PCA UK Yield Curves\\PCA_Spot_Curve.xls"
data=pd.ExcelFile(path)
df1=data.parse('Changes (bps)')
df2=df1.iloc[3:759:,1:51]
df2 = df2.astype(float)
df3 =df2.corr()
egn,egnvec=eig(df3)
df4 = pd.DataFrame(egn)
df5=pd.DataFrame(egnvec)
#mmult=df2.dot(df5)
#converts to numpy array

sptchanges=df2.to_numpy()
eigvect=df5.to_numpy()
pca=np.dot(sptchanges,eigvect)
reverse=np.dot(pca,eigvect.transpose())
y=np.std(sptchanges, axis=0,ddof=1)
D=np.diag(y)
covar=np.dot(D,df3)
covar=np.dot(covar, D)
egncov,egnveccov=eig(covar)
