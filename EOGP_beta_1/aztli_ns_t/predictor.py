from aztli_ns_t import archVAE_T as av
from aztli_ns_t import archMLP_T as am
from aztli_ns_t import readerGraphics_T as rg
import numpy as np
from cv2 import split

def loadNN():
    vae = av.ensembleVAE()
    vae.load_weights('aztli_ns_t/vaet_7_1500.weights.h5')    
    mlp = am.MLP()
    mlp.load_weights('aztli_ns_t/mlp_7PT_1500.weights.h5')
    return mlp, vae

def predictGraphs(X_Test, mlp, vae):        
    zPred = mlp.predict(X_Test)
    Ygraphs = vae.decoder.predict(zPred)
    return Ygraphs

def predictK(cy, Xtest, mlp, vae):
    graphs = predictGraphs(Xtest, mlp, vae)
    alpha = np.array([])
    K = np.array([])
    for i in range(len(graphs)):
        graphCy, graphK = split(graphs[i])
        alpha = np.append(alpha, rg.searchAlphawithCy(cy, graphCy))
        K = np.append(K, rg.searchEwithCy(cy, graphK))
    return K, alpha
