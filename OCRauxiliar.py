# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:26:26 2018

@author: nicot

OCR Auxiliar
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

def convGris(img):
    """
    img: tipo 'uint8' o 'float32', de 3 canales.
    """
    red, green, blue = img[:,:,0], img[:,:,1],img[:,:,2]
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return gray

def mat2img(mat):
    mat = mat.astype("float32")
    return (mat/mat.max()*255).astype("uint8")

def elegirCoord(fotoDif):
    """
    - Key element: setMouseCallback. 
    - Instructions: dameCoordenadas
    Tengo que elegir las 4 esquinas de la imagen 
    También dos puntos cuya distancia en x sea la separación entre dígitos.
    """
    def dameCoordenadas(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y)
            coord.append([x, y])
    tryAgain = True
    while tryAgain:
        coord = []
        cv2.namedWindow('fotoDif', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('fotoDif',dameCoordenadas)
        print("Seleccione:\n \
              1) Borde inferior izquierdo\n\
              2) Borde superior izquierdo\n\
              3) Borde superior derdecho\n\
              4) Borde inferior derecho \n\
              5) Punto en borde de dígito (para saber separación entre dígitos)\n\
              6) Punto en borde de dígito adyacente al anterior")
        while(len(coord) < (4+2)):
            cv2.imshow('fotoDif',fotoDif)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        # Reviso si la región seleccionada es correcta.
        if coord[0][0]<coord[2][0] and coord[0][1]>coord[2][1]:
            tryAgain = False
        else:
            print("Probar seleccionando una región válida.")
    return coord
#%%
def cutROI(imagen,c_t,mostrar=True):
    rows,cols = imagen.shape
    ancho = abs(c_t[3][0]-c_t[0][0])
    alto = abs(c_t[0][1]-c_t[1][1])
    
    pts1 = np.float32(c_t[:4])
    pts2 = np.float32([[0,alto],[0,0],[ancho,0],[ancho,alto]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    output = cv2.warpPerspective(imagen,M,(ancho,alto))

    if mostrar:
        plt.figure(1), plt.suptitle("Set up ROI")
        plt.subplot(121),plt.imshow(imagen, cmap='Greys_r'),plt.title('Input')
        plt.subplot(122),plt.imshow(output, cmap='Greys_r'),plt.title('Output')
        plt.show(), plt.waitforbuttonpress(), plt.close('all')
    return np.array(output, dtype='uint8')

def setupROI(imagenROI, N, c_t, mostrar=True):
    """
    Ahora solo segmenta los digitos. Antes también agarraba el ROI
        pero eso lo hace cutROI
    """
    output = imagenROI
    ancho = imagenROI.shape[1]
    
    digitos = []
    # Espacio entre dígitos
    d = abs(c_t[4][0]-c_t[5][0])
    # Ancho de dígitos
    dx = int((ancho-(N-1)*d)/N)
    for i in range(N):
        digitos.append(output[:,i*(dx+d):(i+1)*dx+i*d])
    
    if mostrar:
        fig_digitos, ax_digitos = plt.subplots(1,N)
        fig_digitos.suptitle("Digitos segmentados.")
        for i in range(N):
            plt.subplot(1,N,i+1)
            plt.imshow(digitos[i], cmap='Greys_r')
        plt.waitforbuttonpress()
    return np.array(digitos, dtype="uint8")
#%%
def binarizar(digitos, adaptive=False, size=151, C=0, mostrar=True):
    """
    Digitos es una matriz con los digitos (en indice 0).
    """
    digitos_bin = []
    
#    size = digitos.shape[1] + (digitos.shape[1]+1)%2
#    print(size)
    for dig in digitos:
        if adaptive:
            digitos_bin.append(cv2.adaptiveThreshold(dig,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,size,C))        
        else:
            digitos_bin.append(cv2.threshold(dig,int(np.mean(dig)),255,cv2.THRESH_BINARY)[1])

    if mostrar:
        N = len(digitos)
        fig_digitos, ax_digitos = plt.subplots(2,N)
        fig_digitos.suptitle("Digitos segmentados y binarización.")
        for i in range(N):
            plt.subplot(2,N,i+1)
            plt.imshow(digitos[i], cmap='Greys_r')
            plt.subplot(2,N,N+i+1)
            plt.imshow(digitos_bin[i], cmap='Greys_r')
        plt.waitforbuttonpress()
    return np.array(digitos_bin, dtype="uint8")
#%%
def binarizarUnaImagen(imagen, adaptive=False, size=151, C=7, mostrar=True):
    """
    Digitos es una matriz con los digitos (en indice 0).
    """
    digitos_bin = []
    binarizada = cv2.adaptiveThreshold(imagen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,size,C)

    if mostrar:
#        N = len(digitos)
        N = 2
        fig_digitos, ax_digitos = plt.subplots(1,2)
        fig_digitos.suptitle("binarizarUnaImagen")
        
        plt.subplot(1,2,1)
        plt.imshow(imagen, cmap='Greys_r')
        plt.subplot(1,2,2)
        plt.imshow(binarizada, cmap='Greys_r')
        plt.waitforbuttonpress()
    return np.array(binarizada, dtype="uint8")
#%%
def suavizarImagen(digitos_bin, pix=4, mostrar=True):
    """
    img: uint8 o float32
    pix: tamaño de la cuadrícula donde "promediar"
    """
    digitos_bin_suav = [] # Lo voy a modificar
    # Defino si la región alrededor (+- pix) de cada punto es negra o blanca
    for n in range(digitos_bin.shape[0]):
        r_x = pix
        r_y = pix
        img = digitos_bin[n,:,:].astype("float32")
        img_2 = (img[:-r_x,:-r_y]+img[r_x:,:-r_y]+img[:-r_x,r_y:]+img[r_x:,r_y:])/4
        for i in range(img_2.shape[0]):
           for j in range(img_2.shape[1]):
               if img_2[i,j] > (128 + 2):
                   img_2[i,j] = 255
               else:
                   img_2[i,j] = 0
        # Erosiono las líneas que me quedaron por usar el método anterior
        erosion_size = 1
        erosion_type = 0
        element = cv2.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
        dil = cv2.dilate(img_2.astype("uint8"),element)

        dil = cv2.resize(dil, dsize=img.shape[::-1], interpolation = cv2.INTER_CUBIC)
        digitos_bin_suav.append(dil)
    if mostrar:
        N = len(digitos_bin)
        fig_digitos, ax_digitos = plt.subplots(2,N)
        fig_digitos.suptitle("Digitos segmentados y suavizado.")
        for i in range(N):
            plt.subplot(2,N,i+1)
            plt.imshow(digitos_bin[i], cmap='Greys_r')
            plt.subplot(2,N,N+i+1)
            plt.imshow(digitos_bin_suav[i], cmap='Greys_r')
        plt.waitforbuttonpress()
#        cv2.imshow('SuavImagen',np.vstack((digitos_bin, digitos_bin_suav)))
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    return np.array(digitos_bin_suav, dtype='uint8')

def CargarBaseReescalar(file_base, digitos_bin, mostrar=True):
    # Cargar base 
    base = cv2.imread(file_base, cv2.IMREAD_GRAYSCALE)
    dx = int(base.shape[1]/10) # Width0
    
    num_base0 = []
    num_base = []
    for i in range(10):
        num_base0.append(base[:,i*dx:(i+1)*dx])
        num_base.append(cv2.resize(num_base0[i],dsize=digitos_bin.shape[1:][::-1], interpolation = cv2.INTER_CUBIC))
    
    if mostrar:
        fig_digitos, ax_digitos = plt.subplots(2,10)
        fig_digitos.suptitle("Base original y escalada.")
        for i in range(10):
            plt.subplot(2,10,i+1)
            plt.imshow(num_base0[i], cmap='Greys_r')
            plt.xticks([]), plt.yticks([])
            plt.subplot(2,10,10+i+1)
            plt.imshow(num_base[i], cmap='Greys_r')
            plt.xticks([]), plt.yticks([])
        plt.waitforbuttonpress()
    return np.array(num_base)

#%% Comparo con la base de datos
def comparar(digitos_bin, num_base, mostrar=True):
    """
    Devuelve 2 matrices:
        Filas: dígito a leer los pesos de coincidencia con cada dígito de menor a mayor.
        Columnas: digitos de la base.
        Valores res_posibles: pesos de cada comparación.
        Valores confianzas: distancias relativas al siguiente valor posible.
    """
#    resultado = [] Este no sirve porque es inlexible, mejor analisis
    analisis = [] # Acá pongo cuánto coincide con cada dígito
    for n in range(len(digitos_bin)):
        pesos = []
        for num in range(10):
            peso_i = 0
            for i in range(digitos_bin.shape[1]):
                for j in range(digitos_bin.shape[2]):
                    if digitos_bin[n,i,j] == num_base[num,i,j]:
                        peso_i += 1
            pesos.append(peso_i)
#        resultado.append(pesos.index(max(pesos)))
        analisis.append(pesos)
    
    # Cuantificacion
    analisis = np.array(analisis)
    # Los numeros posibles orenados de menos probable a más probable
    res_posibles = np.argsort(analisis, axis=1)
    # Los valores de coincidencias correspondientes
    ordenado = np.sort(analisis, axis=1)
    # Armo los pesos con las distancias al siguiente valor
    intervalos = (np.max(analisis, axis=1)-np.min(analisis, axis=1)).reshape(analisis.shape[0],1)
    confianzas = (ordenado[:,1:]-ordenado[:,:-1])*(1/intervalos)
#    print(res_posibles[:,-1])
#    print(np.round(confianzas[:,-1]*100,0))
#    print(res_posibles[:,-2])
#    print(np.round(confianzas[:,-2]*100,0))
    if mostrar:
        plt.figure(), plt.title("Coincidencias por dígito")
        i = 0
        for pesos in analisis:
            plt.plot(pesos, label=str(i))
            i += 1
        plt.legend()
        plt.waitforbuttonpress(), plt.close('all')
    return res_posibles, confianzas

#res_posibles, confianza = comparar(digitos_bin, num_base)
#
#print(res_posibles[:,-1],confianza[:,-1])
#print(res_posibles[:,-2],confianza[:,-2])

#%% Por fracción ocupada de negro
def posibilidadesPorcentaje(digitos_bin, num_base):
    def porcentajeSegmentos(digitos_bin):
        porcentajes = []
        for i in range(digitos_bin.shape[2]):
            fraccionBlanca = np.count_nonzero(digitos_bin[:,:,i])/digitos_bin[:,:,i].size
            porcentajes.append(100*(1-fraccionBlanca))
        return np.array(porcentajes)
    # Porcentaje de los numeros de la base
    porcentaje_base = porcentajeSegmentos(num_base)
    # Porcentajes de cada uno de los digitos de la foto
    porcentaje_dig = porcentajeSegmentos(digitos_bin)
    
    
    posibles_tot = []
    for i in range(digitos_bin.shape[0]):
        posibles = []
        distancias = np.abs(porcentaje_dig[i]-porcentaje_base)
        for j in range(len(distancias)):
            if distancias[j] < 7:
                posibles.append(j)
        posibles_tot.append(posibles)
    
    return posibles_tot
    
#print(posibilidadesPorcentaje(digitos_bin, num_base))

#%% Sub fragmentacion de digito
def fragmentDigitos(digito):
    dx = int(digito.shape[1]/4) # Divisiones de ancho
    dy = int(digito.shape[0]/7) # Divisiones de altura
    
    s1 = digito[(1*dy):(3*dy),:dx] # 2do 5to izquierda
    s2 = digito[:dy,(1*dx):(3*dx)] # 1er 5to arriba
    s3 = digito[(1*dy):(3*dy),(3*dx):(4*dx)] # Segundo quinto derecha
    s4 = digito[(4*dy):(6*dy),(3*dx):(4*dx)] # 4to 5to derecha
    s5 = digito[(6*dy):(7*dy),(1*dx):(3*dx)] # 5to 5to abajo
    s6 = digito[(4*dy):(6*dy),:dx] # 
    s7 = digito[(3*dy):(4*dy),(1*dx):(3*dx)] # 3er 5to centro
    
    lista = [s1,s2,s3,s4,s5,s6,s7]
    porcentajes = []
    for i in range(7):
        fraccionBlanca = np.count_nonzero(lista[i])/lista[i].size
        porcentajes.append(round(fraccionBlanca*100))
    return porcentajes 

#%%
def metodoSegmentos(digitos):
    DIGITS_LOOKUP = [
        [1, 1, 1, 1, 1, 1, 0], # 0
        [0, 0, 1, 1, 0, 0, 0], # 1
        [0, 1, 1, 0, 1, 1, 1], # 2
        [0, 1, 1, 1, 1, 0, 1], # 3
        [1, 0, 1, 1, 0, 0, 1], # 4
        [1, 1, 0, 1, 1, 0, 1], # 5
        [1, 1, 0, 1, 1, 1, 1], # 6
        [0, 1, 1, 1, 0, 0, 0], # 7
        [1, 1, 1, 1, 1, 1, 1], # 8
        [1, 1, 1, 1, 1, 0, 1], # 9
        [0, 0, 0, 0, 0, 0, 1]] # -
    numeros = []
    distancias_num = []
    for i in range(digitos.shape[0]):
        porcentajes_i = fragmentDigitos(digitos[i]) # Lista
        m = np.min(porcentajes_i)
        M = np.max(porcentajes_i)
        distancias = []
        for j in range(10):
            digit_j = np.array(DIGITS_LOOKUP[j])
            map_digit_j = (m-M)*digit_j + M
            dist = np.sum( (np.abs(porcentajes_i-map_digit_j)/(M-m))**2  )/7
            distancias.append(dist)
        orden = np.argsort(distancias)
#        print("Orden de más probable a menos probable para dígito %i :"%i)
#        print(orden)
        numeros.append(orden)
        distancias_num.append(np.round(np.sort(distancias)*100))
    
    numeros = np.array(numeros)
    distancias_num = np.array(distancias_num)
    print(numeros[:,0])
    print(distancias_num[:,0])
    print(numeros[:,1])
    print(distancias_num[:,1])
    
    return numeros, distancias_num

#%% Cargar imagen de cámara web

#cap = cv2.VideoCapture(0) # Objeto Video Capture
def mostrarWebcam(cap):
    print("Mostrando imagen. Capturar con 'q'...")
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # Our operations on the frame come here
    #    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Display the resulting frame
        cv2.imshow('frame',frame)
    #    cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print("Dejo de mostrar")