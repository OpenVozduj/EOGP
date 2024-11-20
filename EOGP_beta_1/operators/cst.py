import numpy as np
from math import factorial
from sklearn.preprocessing import MinMaxScaler

def normParamCST():
    XT = np.load('operators/airfoilsCSTplus.npy')
    scaler = MinMaxScaler(feature_range=(0, 1))
    normXT = scaler.fit_transform(XT)
    return normXT, scaler 

def cstN6(A, N1=0.5, N2=1.0, n=51):
    X = np.zeros(n)
    for i in range(n):
        X[i] = 1-np.cos((i*np.pi)/(2*(n-1)))
    C = X**N1*(1-X)**N2
    AU = A[:7]
    AL = A[7:]
    N = AU.size-1
    SU = np.zeros_like(X)
    for i in range(N+1):
        SU = SU + AU[i]*(factorial(N)/(factorial(i)*factorial(N-i)))*X**i*(1-X)**(N-i)
    SL = np.zeros_like(X)
    for i in range(N+1):
        SL = SL + AL[i]*(factorial(N)/(factorial(i)*factorial(N-i)))*X**i*(1-X)**(N-i)
    YU = C*SU
    YL = C*SL
    return X, YU, YL

def cst_ST_SC(A, N1=0.5, N2=1.0, n=51):
    X = np.zeros(n)
    for i in range(n):
        X[i] = 1-np.cos((i*np.pi)/(2*(n-1)))
    C = X**N1*(1-X)**N2
    AU = A[:7]
    AL = A[7:]
    N = AU.size-1
    ST = np.zeros_like(X)
    for i in range(N+1):
        ST = ST + 0.5*(AU[i]-AL[i])*(factorial(N)/(factorial(i)*factorial(N-i)))*X**i*(1-X)**(N-i)
    SC = np.zeros_like(X)
    for i in range(N+1):
        SC = SC + 0.5*(AU[i]+AL[i])*(factorial(N)/(factorial(i)*factorial(N-i)))*X**i*(1-X)**(N-i)
    YT = 2*C*ST
    YC = C*SC
    return YT, YC