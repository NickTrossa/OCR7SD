# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:59:02 2018

@author: nicot

Analysis of stored photos in ./img folder.

Pending:
    Short-term:
        Add adaptive Threshold config after brightness and contrast
        Improve possible results plot.
        create a function for the case of mostrar=True in OCRauxiliar
    Medium-term:
        auto detect number of digits
        add example for webcam
        use adaptive threshold as default and add backgrund substract as an option
    Long-term:
        try code with other numbers font (other num_base.png)

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

# Set adaptive thresholding parameters
setup.update({"winSize":73,"C":0})
#%%
for n in range(19,52):
    # Load images
    fotoDif = cv2.imread("./tanda1/%i.png"%n, cv2.IMREAD_GRAYSCALE)
    # Process the digits
    
    res_posibles, confianzas = ocr.adquirirNumero(fotoDif, setup, ver=False)
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