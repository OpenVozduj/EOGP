import numpy as np
import pandas as pd
from operators import cst

def initial_Population(NP):
    airfoils = np.load('operators/cst_param.npy')
    set1 = np.arange(1250)
    set2 = np.arange(1250, 1500)
    xsel1 = np.random.choice(set1, int(NP/2), replace=False)
    xsel2 = np.random.choice(set2, int(NP/2), replace=False)
    xsel = np.append(xsel1, xsel2)
    P = airfoils[xsel]
    return P

def suprEqualities(ft):
    for p in range(len(ft)):
        for q in range(len(ft)):
            if p != q:
                if np.array_equal(ft[p], ft[q]) == True:
                    ft[q] = ft[q] + np.random.rand()/1000
    return ft

def nonDominatedSort(Rt, ft):
    Rc = Rt.copy()
    fc = ft.copy()
    D = np.size(Rc, 1)
    Rt_s = np.empty([0, D])
    ft_s = np.empty([0, 2])
    Ft = np.array([])
    F = 0
    while len(fc) > 0:
        if len(fc) > 1:
            d = []
            for p in range(len(fc)):
                Np = 0
                for q in range(len(fc)):
                    if p != q:
                        sp = 0
                        if fc[p,0] >= fc[q,0]:
                            sp += 1
                        if fc[p,1] >= fc[q,1]:
                            sp += 1
                        if sp == 2:
                            Np += 1
                if Np == 0:
                    Rt_s = np.row_stack((Rt_s, Rc[p]))
                    ft_s = np.row_stack((ft_s, fc[p]))
                    Ft = np.append(Ft, F)
                    d.append(p)
            if len(d) == 0:
                Rt_s = np.row_stack((Rt_s, Rc))
                ft_s = np.row_stack((ft_s, fc))
                Ft = np.append(Ft, F)
                print('F'+str(F)+' '+str(len(fc))+' elements')
                Rc = np.empty([0, D])
                fc = np.empty([0, D])
            else:
                Rc = np.delete(Rc, d, axis=0)
                fc = np.delete(fc, d, axis=0)
                print('F'+str(F)+' '+str(len(d))+' elements')
                F += 1
        else:
            Rt_s = np.row_stack((Rt_s, Rc))
            ft_s = np.row_stack((ft_s, fc))
            Ft = np.append(Ft, F)
            print('F'+str(F)+' '+str(len(fc))+' elements')
            Rc = np.empty([0, D])
            fc = np.empty([0, D])
    return Rt_s, ft_s, Ft

def crowDist(Rt_s, ft_s, Ft):
    D = np.size(Rt_s, 1)
    Rt_sI = np.empty([0, D])
    ft_sI = np.empty([0, 2])
    Idt = np.array([])
    for i in range(int(max(Ft))+1):
        Fi = np.where(Ft==i)[0]
        Rt_Fi = Rt_s[Fi]
        ft_Fi = ft_s[Fi]
        FiGroup = np.column_stack((ft_Fi, Rt_Fi))
        Fidf = pd.DataFrame(FiGroup)
        Fidf = Fidf.sort_values(by=0, ascending=True)
        FiGroup_sort = Fidf.to_numpy()
        Id = np.zeros(len(Fi))
        if Fi.size>2:
            Id[0] = 100
            Id[Fi.size-1] = 100
            for l in range(1, Fi.size-1):
                sm = 0
                for m in range(2):
                    if max(FiGroup_sort[:,m])-min(FiGroup_sort[:,m]) !=0:
                        sm += abs(FiGroup_sort[l+1,m]-FiGroup_sort[l-1,m])/(max(FiGroup_sort[:,m])-min(FiGroup_sort[:,m]))
                Id[l] = sm
        FiGroup_sort = np.column_stack((Id, FiGroup_sort))
        Fidf2 = pd.DataFrame(FiGroup_sort)
        Fidf2 = Fidf2.sort_values(by=0, ascending=False)
        FiGroup_sortI = Fidf2.to_numpy()
        Idt = np.append(Idt, FiGroup_sortI[:,0])
        ft_sI = np.row_stack((ft_sI, FiGroup_sortI[:,1:3]))
        Rt_sI = np.row_stack((Rt_sI, FiGroup_sortI[:,3:]))
        
    return Rt_sI, ft_sI, Idt

def tournament(N, pt=0.8):
    xr = np.random.choice(np.arange(N), 2, replace=False)
    r = np.random.rand()
    if r < pt:
        if xr[0] < xr[1]:
            p = xr[0]
        else:
            p = xr[1]
    else:
        if xr[0] < xr[1]:
            p = xr[1]
        else:
            p = xr[0]
    return p

def SBX(Ptg, N, scaler, eta_c=20):
    D = np.size(Ptg, 1)
    C = np.empty([0, D])
    for i in range(int(N/2)):
        while 2==2:
            while 1==1:
                p1 = tournament(N)
                p2 = tournament(N)
                if p1!=p2:
                    break
            P1 = Ptg[p1]
            P2 = Ptg[p2]
            C1 = np.zeros_like(P1)
            C2 = np.zeros_like(P2)
            for j in range(D):
                u = np.random.uniform()
                if u < 0.5:
                    beta = (2*u)**(1/(eta_c+1))
                else:
                    beta = 1/((2*(1-u))**(1/(eta_c+1)))
                C1[j] = 0.5*((1-beta)*P1[j] + (1+beta)*P2[j])
                C2[j] = 0.5*((1+beta)*P1[j] + (1-beta)*P2[j])
            A1 = scaler.inverse_transform(C1.reshape(1, -1))
            A2 = scaler.inverse_transform(C2.reshape(1, -1))
            yt1, _ = cst.cst_ST_SC(A1[0])
            yt2, _ = cst.cst_ST_SC(A2[0])
            if np.where(yt1<0)[0].size == 0 and np.where(yt2<0)[0].size == 0:
                break
        C = np.row_stack((C, C1, C2))
    return C

def mutation(C, N, Omega, scaler, eta_m=20):
    D = np.size(C, 1)
    Q = np.empty([0, D])
    for i in range(N):
        p = C[i]
        while 1==1:
            m = np.zeros_like(p)
            for j in range(D):
                r = np.random.uniform()
                if r < 0.5:
                    delta = (2*r)**(1/(eta_m+1))-1
                else:
                    delta = 1 - (2*(1-r))**(1/(eta_m+1))
                m[j] = p[j] + delta*(Omega[j,1] - Omega[j,0])
            m = np.clip(m, Omega[:,0], Omega[:,1])
            Am = scaler.inverse_transform(m.reshape(1, -1))
            yt, _ = cst.cst_ST_SC(Am[0])
            if np.where(yt<0)[0].size == 0:
                break
        Q = np.row_stack((Q, m))
    return Q
    