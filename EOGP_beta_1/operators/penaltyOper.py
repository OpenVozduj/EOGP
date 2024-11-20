import numpy as np
from sklearn.preprocessing import MinMaxScaler

def distanceX(E, K, ytmax, ytmaxt, gxType, delta):
    #%% Normalize functions
    f1 = 1/E
    f2 = 1/K
    f = np.column_stack((f1, f2))
    normalizer = MinMaxScaler(feature_range=(0, 1))
    fn = normalizer.fit_transform(f)
    #%% Select restriction
    if gxType==0:
        g = ytmaxt - ytmax
    else:
        g = abs(ytmaxt - ytmax) - delta
    #%% Constrain violation
    c = np.array([])
    for i in range(len(f1)):
        c = np.append(c, max(0, g[i]))
    if max(c) == 0:
        v = np.zeros_like(f1)
    else:
        v = np.array([])
        for i in range(len(f1)):
            v = np.append(v, c[i]/max(c))
    #%% distance
    rf = len(np.where(v==0)[0])/len(f1)
    d = np.zeros_like(fn)
    for i in range(len(f1)):
        for j in range(2):
            if rf == 0:
                d[i,j] = v[i]
            else:
                d[i,j] = np.sqrt(fn[i,j]**2+v[i]**2)
    return fn, v, d

def twoPenalties(fn, v):
    rf = len(np.where(v==0)[0])/len(fn)
    X = np.array([])
    for i in range(len(fn)):
        if rf==0:
            X = np.append(X, 0)
        else:
            X = np.append(X, v[i])
    Y = np.zeros_like(fn)
    for i in range(len(fn)):
        for j in range(2):
            if v[i] == 0:
                Y[i,j] = 0
            else:
                Y[i,j] = fn[i,j]
    p = np.zeros_like(fn)
    for i in range(len(fn)):
        for j in range(2):
            p[i,j] = (1-rf)*X[i]+rf*Y[i,j]
    return p

def fitnessPen(E, K, ytmax, ytmaxt, gxType=0, delta=0.004):
    fn, v, d = distanceX(E, K, ytmax, ytmaxt, gxType, delta)
    p = twoPenalties(fn, v)
    ft = p + d
    return ft

