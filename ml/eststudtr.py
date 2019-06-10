#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eststudtr.py

Purpose:
    Estimate a student-t regression model, using lambda function, with extra restrictions, using transformations

Version:
    1       Following estnorm.py/estcointr2.py

Date:
    2017/8/21, 2018/12/6

Author:
    Charles Bos
"""
###########################################################
### Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as st
import math

###########################################################
### Get hessian and related functions
from lib.grad import *
from lib.printtex import *
    
###########################################################
def GetPars(vP):
    """
    Purpose:
      Read out the parameters from the vector

    Inputs:
      vP        iK+2 vector with nu, sigma and beta's

    Return value:
      dNu       double, degrees of freedom
      dS        double, sigma
      vBeta     iK vector, beta's
    """
    iK= np.size(vP)-2
    # Force vP to be a 1D matrix
    vP= vP.reshape(iK+2,)
    dNu= vP[0]
    dS= vP[1]
    vBeta= vP[2:]

    return (dNu, dS, vBeta)

###########################################################
def TransPar(vP):
    """
    Purpose:
      Transform the parameters from the vector

    Inputs:
      vP        iK+2 vector with nu, sigma and beta's

    Return value:
      vPTr      iK+1 vector implementing restriction that nu > 2, sigma > 0, beta1+beta2= 1
    """
    iK= np.size(vP)-2
    if (iK < 3):
        print ("Error: Expected at least 3 betas...")

    (dNu, dS, vBeta)= GetPars(vP)
    dNuTr= np.log(dNu-2)
    dSTr= np.log(dS)
    vBetaTr= vBeta[:2]      # Get beta0 and beta1
    if (iK > 3):            # Possibly add in beta 3 and further, skipping beta2
        vBetaTr= np.hstack((vBetaTr, vBeta[3:]))
    vPTr= np.hstack((dNuTr, dSTr, vBetaTr))

    return vPTr

###########################################################
def TransBackPar(vPTr):
    """
    Purpose:
      Transform the parameters back from the vector

    Inputs:
      vPTr      iK+1 vector with nutr, sigmatr and beta's

    Return value:
      vP        iK+2 vector implementing restriction that nu > 2, sigma > 0, beta1+beta2= 1
    """
    iK= np.size(vPTr)-1

    (dNuTr, dSTr, vBetaTr)= GetPars(vPTr)
    dNu= np.exp(dNuTr)+2
    dS= np.exp(dSTr)
    dBeta2= 1-vBetaTr[1]
    vBeta= np.hstack((vBetaTr[:2], dBeta2))      # Get beta0, beta1, beta2
    if (iK > 3):            # Possibly add in beta 3 and further, skipping beta2
        vBeta= np.hstack((vBeta, vBetaTr[2:]))
    vP= np.hstack((dNu, dS, vBeta))

    return vP

###########################################################
def GetParNames(iK):
    """
    Purpose:
      Construct names for the parameters from the vector

    Inputs:
      iK        integer, number of beta's

    Return value:
      asP       iK array, with strings "nu", "sigma", "b1", ...
    """
    asP= ["B"+str(i+1) for i in range(iK)]
    asP= ["Nu", "Sigma"] + asP

    return asP

###########################################################
def GenrX(iN, iK):
    """
    Purpose:
      Generate regressors, constant + uniforms

    Inputs:
      iN        integer, number of observations
      iK        integer, number of regressors

    Return values:
      mX        iN x iK matrix of regressors, constant + uniforms
    """
    mX= np.hstack([np.ones((iN, 1)), np.random.rand(iN, iK-1)])

    return mX
###########################################################
def GenrY(vP, mX):
    """
    Purpose:
      Generate regression data

    Inputs:
      vP        iK+2 vector of parameters, nu, sigma and beta
      mX        iN x iK matrix of regressors

    Return values:
      vY        iN vector of data
    """
    iN= mX.shape[0]
    (dNu, dS, vBeta)= GetPars(vP)
    vY= mX@vBeta + dS * st.t.rvs(dNu, size=iN)

    return vY

###########################################################
### vLL= LnLRegrStud(vP, vY, mX)
def LnLRegrStud(vP, vY, mX):
    """
    Purpose:
        Compute loglikelihood of regression model

    Inputs:
        vP      iK+2 vector of parameters, nu, sigma and beta
        vY      iN 1D-vector of data
        mX      iN x iK matrix of regressors

    Return value:
        vLL     iN vector, loglikelihood contributions
    """
    (iN, iK)= mX.shape
    if (np.size(vP) != iK+2):         # Check if vP is as expected
        print ("Warning: wrong size vP= ", vP)

    (dNu, dSigma, vBeta)= GetPars(vP)
    if (dSigma <= 0):
        print ("x", end="")
        return -math.inf
    if (dNu <= 2):
        print ("t", end="")
        return -math.inf

    vE= vY - mX @ vBeta
    vLL= st.t.logpdf(vE/dSigma, df= dNu) - np.log(dSigma)

    print (".", end="")             # Give sign of life

    return vLL

###########################################################
### (vP, vS, dLL, sMess)= EstimateRegrStud(vY, mX)
def EstimateRegrStud(vY, mX):
    """
    Purpose:
      Estimate the student regression model

    Inputs:
      vY        iN vector of data
      mX        iN x iK matrix of regressors

    Return value:
      vP        iK+2 vector of optimal parameters nu, sigma and beta's
      vS        iK+2 vector of standard deviations
      dLL       double, loglikelihood
      sMess     string, output of optimization
    """
    (iN, iK)= mX.shape
    vP0= np.ones(iK+2)        # Get (bad...) starting values
    vP0[0]= 5                 # Set nu sufficiently high
    vP0[3:5]= np.array((2, -1))          # Make sure I set starting values conforming with the restriction

    vP0Tr= TransPar(vP0)
    vP1= TransBackPar(vP0Tr)
    if (np.max(np.fabs(vP0 - vP1)) > 1e-3):
        print ("Something wrong in the transformation?", vP0, vP1)

    # Create lambda function returning NEGATIVE AVERAGE LL, as function of vPTr only
    AvgNLnLRegrStudTr= lambda vPTr: -np.average(LnLRegrStud(TransBackPar(vPTr), vY, mX))

    dLL= -iN*AvgNLnLRegrStudTr(vP0Tr)
    print ("Initial LL= ", dLL, "\nvP0=", vP0, "vP0Tr= ", vP0Tr)

    res= opt.minimize(AvgNLnLRegrStudTr, vP0Tr, method="BFGS")

    vPTr= res.x
    vP= TransBackPar(vPTr)
    sMess= res.message
    dLL= -iN*res.fun
    print ("\nBFGS results in ", sMess, "\nPars: ", vP, "\nLL= ", dLL, ", f-eval= ", res.nfev)

    # Note: mHq is the hessian of the Average Negative LnL
    mHq= hessian_2sided(AvgNLnLRegrStudTr, vPTr)
    mHtr= -mHq*iN                        # Get true hessian of LnL
    mS2tr= np.linalg.inv(-mHtr)    # Get covariance matrix of vPTr
    # mJ= (np.exp(vPTr)/np.square(1+np.exp(vPTr))).reshape(1,1)   # Jacobian
    mJ= jacobian_2sided(TransBackPar, vPTr)
    mS2= mJ@mS2tr@mJ.T                    # Get covariance matrix
    vS= np.sqrt(np.diag(mS2))   # Get (single) standard deviation

    return (vP, vS, dLL, sMess)

###########################################################
### ShowData(vY, mX)
def ShowData(vY, mX):
    """
    Purpose:
      Provide data plot on screen
    """
    (iN, iK)= mX.shape

    plt.figure(figsize=(8,4))
    iBar= min(max(5, iN/20), 20)
    plt.subplot(1, iK, 1)
    plt.hist(vY, bins=iBar, density=True, label='y')
    plt.legend()
    for i in range(1, iK):
        plt.subplot(1, iK, i+1)
        plt.plot(mX[:,i], vY, 'x', label='y x x_%i' % (i+1))
        plt.legend()

    plt.savefig('graphs/studregrn%ik%i.png' % (iN, iK))
    plt.show()

###########################################################
### Output(mPPS, dLL, sMess, iN)
def Output(mPPS, dLL, sMess, iN):
    """
    Purpose:
      Provide output on screen
    """
    iK= mPPS.shape[1]-2
    print ("\n\nEstimation resulted in ", sMess)
    print ("Using ML with LL= %g over n=%i observations" % (dLL, iN))

    print ("Parameter estimates:")
    printtex(mPPS.T, index=GetParNames(iK), columns=["PTrue", "PHat", "s(P)"], formats="%.3g")

    vBeta= mPPS[1,2:]
    print ("Difference between restriction for b1+b2=1: ", vBeta[1]+vBeta[2]-1)

###########################################################
### main
def main():
    vP0= [4, .1, 1, .8, .2]    # Nu, dSigma and vBeta together
    iN= 500
    iSeed= 1234

    #Generate data
    np.random.seed(iSeed)
    vP0= np.array(vP0)

    iK= vP0.size - 2
    mX= GenrX(iN, iK)
    vY= GenrY(vP0, mX)
    ShowData(vY, mX)

    (vP, vS, dLnPdf, sMess)= EstimateRegrStud(vY, mX)
    Output(np.vstack([vP0, vP, vS]), dLnPdf, sMess, iN);

###########################################################
### start main
if __name__ == "__main__":
    main()
