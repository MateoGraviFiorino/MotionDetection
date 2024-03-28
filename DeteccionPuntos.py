import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def imshow(img, nueva_figura=True, titulo=None, img_a_color=False, bloqueante=True, barra_de_color=False, sin_ticks=False):
    print(img.shape)
    if nueva_figura:
        plt.figure()
    if img_a_color:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(titulo)
    if not sin_ticks:
        plt.xticks([]), plt.yticks([])
    if barra_de_color:
        plt.colorbar()
    if nueva_figura:
        plt.show(block=bloqueante)



#Cargar imagen y pasarla a gris
img1 = cv.imread("img4.jpeg")
img2 = cv.imread("img1.jpeg")

image = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
image2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Iniciar el detector SIFT
sift = cv.SIFT_create()

# Encontrar los keypoints y descriptores con SIFT
kp1, des1 = sift.detectAndCompute(image, None)
kp2, des2 = sift.detectAndCompute(image2, None)

# Parámetros FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # o pasar un diccionario vacío

# Iniciar FLANN matcher
flann = cv.FlannBasedMatcher(index_params, search_params)

# Realizar emparejamientos kNN
matches = flann.knnMatch(des1, des2, k=2)

# Necesitamos dibujar solo los buenos emparejamientos, por lo que creamos una máscara
matchesMask = [[0, 0] for i in range(len(matches))]

# Test de proporción según el artículo de Lowe (https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/lowe_ijcv2004.pdf)
for i, (m, n) in enumerate(matches):
    if m.distance < 0.55 * n.distance:
        matchesMask[i] = [1, 0]

# Parámetros de dibujo para los emparejamientos
draw_params = dict(matchColor=(0, 255, 0),  # color de los emparejamientos
                   singlePointColor=(255, 0, 0),  # color de los puntos individuales
                   matchesMask=matchesMask,  # máscara para seleccionar emparejamientos
                   flags=cv.DrawMatchesFlags_DEFAULT)

# Dibujar los emparejamientos kNN
img3 = cv.drawMatchesKnn(image, kp1, image2, kp2, matches, None, **draw_params)

# Mostrar la imagen resultante
plt.imshow(img3), plt.show()


