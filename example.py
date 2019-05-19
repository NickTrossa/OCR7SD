# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:59:02 2018

@author: nicot

Análisis de tanda de fotos.
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from OCR import configCamara, adquirirNumero, configImagen, adquirirImagen

#%% MAIN BODY

plt.close('all')
setup = configImagen("./img/14.png")

res = np.array([])
res2 = np.array([])
con = np.array([])

#%%
for n in range(14,16):
    
    fotoDif = cv2.imread("./img/%i.png"%n, cv2.IMREAD_GRAYSCALE)
    res_posibles, confianzas = adquirirNumero(fotoDif, *setup, size=73)
    res = np.append(res,res_posibles[:,-1])
    res2 = np.append(res,res_posibles[:,-1])
    con = np.append(con,(confianzas[:,-1]*100).astype("int32"))
    print(res_posibles[:,-3:])
    print((confianzas[:,-3:]*100).astype("int32"))
    print("")
    
    cv2.destroyAllWindows()
    plt.close('all')

cv2.destroyAllWindows()
plt.close('all')