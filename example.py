# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:59:02 2018

@author: nicot

Análisis de tanda de fotos.

Acaso esto se agregó al master o al branch? Al branch. Pero ahora:
    si mergeo se unifica con los cambios hechos al master? Sí. Esto es mágico.
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import cv2

import OCR as ocr

#%% MAIN BODY

plt.close('all')

# Execute configuration part.
setup = ocr.configImagen("./img/14.png")

# A place to hold results 
res = np.array([])
con = np.array([])

#%%
for n in range(14,16):
    # Load images
    fotoDif = cv2.imread("./img/%i.png"%n, cv2.IMREAD_GRAYSCALE)
    # Process the digits
    res_posibles, confianzas = ocr.adquirirNumero(fotoDif, *setup, size=73)
    # Append results and show
    res = np.append(res, res_posibles[:,-1])
    con = np.append(con, (confianzas[:,-1]*100).astype("int32"))
    print("Possible results: digits in rows, decreasing order")
    print(*res_posibles[:,-3:].transpose()[::-1], sep='\n')
    print("Confidence (distance to next option)")
    print(*((confianzas[:,-3:]*100).astype("int32")).transpose()[::-1], sep='\n')
    print("")
    
    cv2.destroyAllWindows()
    plt.close('all')

cv2.destroyAllWindows()
plt.close('all')