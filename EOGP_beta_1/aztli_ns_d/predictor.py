from aztli_ns_d import archVAE_D as av
from aztli_ns_d import archMLP_D as am
from aztli_ns_d import readerGraphics_D as rg
import numpy as np
from cv2 import split

def loadNN():
    vae = av.ensembleVAE()
    vae.load_weights('aztli_ns_d/vaes_7_1500.weights.h5')    
    mlp = am.MLP()
    mlp.load_weights('aztli_ns_d/mlp_7P_1500.weights.h5')
    return mlp, vae

def predictGraphs(X_Test, mlp, vae):        
    zPred = mlp.predict(X_Test)
    Ygraphs = vae.decoder.predict(zPred)
    return Ygraphs

def predictMaxE(Xtest, mlp, vae):
    graphs = predictGraphs(Xtest, mlp, vae)
    E = np.array([])
    alpha = np.array([])
    for i in range(len(graphs)):
        _, _, graphE = split(graphs[i])
        e, aoa = rg.searchMaxE(graphE)
        E = np.append(E, e)
        alpha = np.append(alpha, aoa)
    return E, alpha
