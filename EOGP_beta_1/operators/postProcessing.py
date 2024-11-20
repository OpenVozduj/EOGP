import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
from operators import cst
from sklearn.cluster import KMeans

def showFitnessSort(ft_s, Ft, nameTest, nfig=1):
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 11}
    cm=1/2.54
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)  
    
    f0 = ft_s[:,0]
    f1 = ft_s[:,1]
    
    fig = plt.figure(nfig, figsize=(9*cm, 9*cm))
    ax = fig.add_subplot(111)
    colors = ['black', 'grey', 'lightgrey', 'darkred', 'red', 'salmon',
              'chocolate', 'sandybrown', 'tan', 'moccasin',
              'goldenrod', 'gold', 'yellow', 'darkgreen', 'yellowgreen',
              'lawngreen', 'lightseagreen', 'turquoise', 'cyan', 'darkblue',
              'blue', 'skyblue', 'darkviolet', 'magenta', 'violet']
    k = 0
    for i in range(int(max(Ft))+1):
        ax.scatter(f0[np.where(Ft==i)[0]], 
                   f1[np.where(Ft==i)[0]], c=colors[k], label='F'+str(i))
        k += 1
        if k>24:
            k = 0
    ax.set_xlabel('$F_K$')
    ax.set_ylabel('$F_E$')
    ax.grid()
    # ax.legend(loc='best')
    plt.tight_layout()
    fig.savefig('postProcess/'+nameTest+'_Fitness_'+str(nfig-1)+'.png')
    plt.close()
    
def showEK(E, K, nameTest, nfig=1):
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 11}
    cm=1/2.54
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)  
    
    fig = plt.figure(500+nfig, figsize=(9*cm, 9*cm))
    ax = fig.add_subplot(111)
    ax.scatter(K, E, c='red')
    ax.set_xlabel('$(c_y/c_x)_t$')
    ax.set_ylabel('$(c_y^{1.5}/c_x)_s$')
    ax.grid()
    plt.tight_layout()
    fig.savefig('postProcess/'+nameTest+'_objFunctions_'+str(nfig-1)+'.png')
    plt.close()

def sortFrontPareto(E, K, P):
    D = np.column_stack((E, K))
    D = np.column_stack((D,P))
    Ddf = pd.DataFrame(D)
    Ddf = Ddf.sort_values(by=0, ascending=True)
    D_sort = Ddf.to_numpy()
    fpareto = D_sort[:,:2]
    Ppareto = D_sort[:,2:]
    return Ppareto, fpareto

def showFrontPareto(fpareto, nameTest, K=3, nfig=1000):
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 11}
    cm=1/2.54
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)  

    kmeans = KMeans(n_clusters=K, random_state=0).fit(fpareto)
    kLabels = kmeans.labels_
    kCenters = kmeans.cluster_centers_

    f0 = fpareto[:,0]
    f1 = fpareto[:,1]
    
    fig = plt.figure(nfig, figsize=(9*cm, 9*cm))
    ax = fig.add_subplot(111)
    ax.plot(f1, f0, '-k', label='Фронт Парето')
    
    colors = ['green', 'blue', 'red']
    for k in range(K):
        ax.scatter(f1[np.where(kLabels==k)[0]], 
                   f0[np.where(kLabels==k)[0]], 
                   c=colors[k], label='Скопление_'+str(k))
    
    ax.set_xlabel('$(c_y/c_x)_t$')
    ax.set_ylabel('$(c_y^{1.5}/c_x)_s$')
    ax.grid()
    ax.legend(loc='best')
    plt.tight_layout()
    fig.savefig('postProcess/'+nameTest+'_frontPareto.png')
    plt.close()
    return kLabels
        
    
def optimalAirfoils(A, kLabels, nameTest):
    dirAirfoil = 'postProcess/airfoils_'+nameTest
    os.mkdir(dirAirfoil)
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 11}
    cm=1/2.54
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)  
    
    for i in range(len(A)):
        X, YU, YL = cst.cstN6(A[i])
        fig = plt.figure(i+1, figsize=(9*cm, 3*cm))
        ax = fig.add_subplot(111)
        if kLabels[i]==0:
            ax.plot(X, YU, '-g')
            ax.plot(X, YL, '-g')
        elif kLabels[i]==1:
            ax.plot(X, YU, '-b')
            ax.plot(X, YL, '-b')
        else:
            ax.plot(X, YU, '-r')
            ax.plot(X, YL, '-r')
        ax.set_xlabel('x/b')
        ax.set_ylabel('y/b')
        ax.grid()
        ax.set_aspect('equal')
        plt.tight_layout()
        fig.savefig(dirAirfoil+'/airfoil_'+str(i)+'.png')
        plt.close()
        
    
    
