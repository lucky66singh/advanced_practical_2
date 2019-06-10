#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
printtex.py

Purpose:
    Print a latex matrix

Version:
    1       First start
    2       Allowing for complex values

Date:
    2017/9/29, 2018/11/13

Author:
    Charles Bos
"""
###########################################################
### Imports
import numpy as np
import pandas as pd

###########################################################
### printtex_error(mX, index=None, columns=None, formats=None):
def printtex_error(mX, index=None, columns=None, formats=None):
    """
    Purpose:
        Print a latex matrix, with column formatting

    Inputs:
        mX      iN x iK matrix of data
        index   iN array of row names
        columns iK array of column names
        formats iK array of column formats
    """

    asF1= None
    if (not formats is None):
        # asF1= [ lambda x: sForm % x for sForm in formats]
        iK= len(formats)
        asF1= []
        for i in range(iK):
            fn= lambda x: formats[i] % x
            asF1= asF1 + [fn]
            print ("Testing format %i: %s" % (i, fn(np.random.randn())))

    df= pd.DataFrame(mX, index=index, columns=columns)
    print (df.to_latex(formatters=asF1, escape=False))

    print ("Not functioning... Something weird is going on with python memory?")

###########################################################
### printtex(mX, index=None, columns=None, formats="%.6g"):
def printtex(mX, index=None, columns=None, formats="%.6g", bTex=True):
    """
    Purpose:
        Print a latex matrix, with column formatting

    Inputs:
        mX      iN x iK matrix of data; a iK vector is considered a row vector
        index   iN array of row names
        columns iK array of column names
        formats single format string, or iK array of column formats, default= "%.6g"
        bTex    boolean, default= True. If False, print without LaTeX markers

    Author:
        Charles Bos
    """
    sSep= " & "
    sLine= " \\\\\n"
    if (not bTex):
        sSep= " "
        sLine= " \n"
    if (mX.ndim == 1):
        mX.shape= (1, len(mX))
    [iN, iK]= mX.shape

    if (isinstance(formats, str)):
        formats= [formats]*iK
    if (isinstance(columns, str)):
        columns= [columns]*iK
    if (isinstance(index, str)):
        index= [index]*iN
    # if (not index == None)
    #     index= list(index)

    bInd= (not index == None) and (len(index) >= iN)
    bCol= (not columns == None) and (len(columns) >= iK)

    bFor= len(formats) >= iK

    iWl= 0
    if (bInd):
        vWl= [len(index[i]) for i in range(iN)]
        iWl= max(vWl)

    if (bCol):
        if (bInd):
            print (" "*iWl, sSep, end='')
        for j in range(iK):
            print (columns[j], sSep if j < iK-1 else sLine, end='')

    sFmt= "%" + "%is" % iWl
    for i in range (iN):
        if (bInd):
            print (sFmt % index[i], sSep, end= "")
#            print ("Printint index in format %is..." % iWL)
        for j in range(iK):
            if (np.logical_or(np.isnan(mX[i,j]), mX[i,j] is None)):
                print (" ", end='')
            else:
                if (np.iscomplex(mX[i,j])):
                    sFmtI= '+'+formats[j]+'j' if (mX[i,j].imag > 0) else formats[j]+'j'
                    print (formats[j] % mX[i,j].real, sFmtI % mX[i,j].imag, end='', sep='')
                else:
                    print (formats[j] % mX[i,j].real, end='')
            print (sSep if j < iK-1 else sLine, end='')

###########################################################
### printmat(mX, index=None, columns=None, formats="%.6g"):
def printmat(mX, index=None, columns=None, formats="%.6g"):
    """
    Purpose:
        Print a matrix, with column formatting, without LaTeX

    Inputs:
        mX      iN x iK matrix of data; a iK vector is considered a row vector
        index   iN array of row names
        columns iK array of column names
        formats single format string, or iK array of column formats, default= "%.6g"

    Author:
        Charles Bos
    """
    printtex(mX, index=index, columns=columns, formats=formats, bTex=False)

###########################################################
### main
def main():
    # Test, create fake data
    iN= 4
    mX= np.hstack([np.random.randn(iN, 1), np.fabs(np.random.randn(iN, 1))])
    mX= np.hstack([mX, mX[:,0:1]/mX[:,1:2], np.random.rand(iN, 1)])

    asCol= ["beta", "s(beta)", "t(beta)", "p-val"]
    asRow= list("ABCD")

    asFmt= ["%6.3f", "(%.2f)", "%6.3f", "[%.4f]"]

    print ("No formatting:")
    printtex(mX)
    print ("With formatting:")
    printtex(mX, formats= asFmt, columns= asCol, index= asRow)
    print ("With formatting, no latex:")
    printtex(mX, formats= asFmt, columns= asCol, index= asRow, bTex=False)

###########################################################
### start main
if __name__ == "__main__":
    main()
