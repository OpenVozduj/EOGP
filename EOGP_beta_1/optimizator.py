import numpy as np
from aztli_ns_d import predictor as prd
from aztli_ns_t import predictor as prt
from operators import cst
from operators import geneticOper as go
from operators import penaltyOper as po
from operators import postProcessing as pp
import time
#%%
t_start = time.time()

#%% Initialization
_, scaler = cst.normParamCST()

mlpD, vaeD = prd.loadNN()
mlpT, vaeT = prt.loadNN()

cyt = 0.2
ytmaxt = 0.12
G = 100
N = 150
nameTest = 'gp1202_1'
Omega = np.array([[0.3, 1.0], #A0
                  [0.2, 1.0], #A1
                  [0.2, 1.0], #A2
                  [0.2, 1.0], #A3
                  [0.2, 1.0], #A4
                  [0.2, 1.0], #A5
                  [0.2, 1.0], #A6
                  [0.0, 0.5], #A7
                  [0.0, 0.5], #A8
                  [0.0, 0.6], #A9
                  [0.0, 0.6], #A10
                  [0.0, 0.6], #A11
                  [0.0, 0.7], #A12
                  [0.0, 0.7]]) #A13
saveG = [19, 39, 59, 79, 99]

#%% Initial population

Ptc = go.initial_Population(N)
Pt = scaler.transform(Ptc)

E, _ = prd.predictMaxE(Pt, mlpD, vaeD)
K, _ = prt.predictK(cyt, Pt, mlpT, vaeT)

ytmax = np.array([])
for i in range(N):
    ytmax = np.append(ytmax, max(cst.cst_ST_SC(Ptc[i])[0]))
    
#%% Define fitness with penalties

ft = po.fitnessPen(E, K, ytmax, ytmaxt, gxType=1)
ft = go.suprEqualities(ft)

#%% Sorting

Pt_s, ft_s, Ft = go.nonDominatedSort(Pt, ft)
Ptg, ftg, Idt = go.crowDist(Pt_s, ft_s, Ft)
pp.showFitnessSort(ftg, Ft, nameTest)
pp.showEK(E, K, nameTest)
print('Generation 0')

#%% Loop

for g in range(1,G):
    Ctg = go.SBX(Ptg, N, scaler)
    Qtg = go.mutation(Ctg, N, Omega, scaler)
    
    Rtg = np.row_stack((Ptg, Qtg))
    
    E, _ = prd.predictMaxE(Rtg, mlpD, vaeD)
    K, _ = prt.predictK(cyt, Rtg, mlpT, vaeT)
    Rtgc = scaler.inverse_transform(Rtg)
    ytmax = np.array([])
    for i in range(2*N):
        ytmax = np.append(ytmax, max(cst.cst_ST_SC(Rtgc[i])[0]))
    
    ft = po.fitnessPen(E, K, ytmax, ytmaxt, gxType=1)
    ft = go.suprEqualities(ft)
    
    Rtg_s, ft_s, Ft = go.nonDominatedSort(Rtg, ft)
    Rtg_sI, ft_sI, Idt = go.crowDist(Rtg_s, ft_s, Ft)
    
    if g in saveG:
        pp.showFitnessSort(ft_sI, Ft, nameTest, nfig=g+1)
        pp.showEK(E, K, nameTest, nfig=g+1)
    
    Ptg = Rtg_sI[:N]
    ftg = ft_sI[:N]
    print('Generation '+str(g))

#%% Pareto front
Ftg = Ft[:N]
Pp = Ptg[np.where(Ftg==0)[0]]
Ep, _ = prd.predictMaxE(Pp, mlpD, vaeD)
Kp, _ = prt.predictK(cyt, Pp, mlpT, vaeT)
Ppareto, fpareto = pp.sortFrontPareto(Ep, Kp, Pp)
Pparetoc = scaler.inverse_transform(Ppareto)

kLabels = pp.showFrontPareto(fpareto, nameTest)
pp.optimalAirfoils(Pparetoc, kLabels, nameTest)

np.save('postProcess/Ppareto_'+nameTest+'.npy', Ppareto)
np.save('postProcess/fpareto_'+nameTest+'.npy', fpareto)

delta_t = time.time() - t_start
print(delta_t)