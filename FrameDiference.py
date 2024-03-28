import cv2
import numpy as np

# Función para dibujar contornos en una imagen
def draw_contours(image, contours):
    result_image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return result_image

# Función para procesar el fotograma y detectar movimientos
def process_frame_difference_full(new_image, prev_image, min_area_threshold=100, **kwargs):
    # Convertir las imágenes a escala de grises
    new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

    # Calcular la diferencia absoluta entre los fotogramas actual y anterior
    frame_diff = cv2.absdiff(new_gray, prev_gray)

    # Normalizar la imagen de diferencia
    norm_diff = cv2.normalize(frame_diff, None, 0, 255, cv2.NORM_MINMAX)

    # Umbralizar la imagen para resaltar las diferencias
    _, thresh = cv2.threshold(norm_diff, 30, 255, cv2.THRESH_BINARY)

    # Dilatar la imagen umbralizada para mejorar la detección de contornos
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Convertir la imagen dilatada a formato adecuado para findContours
    dilated = dilated.astype(np.uint8)

    # Encontrar contornos en la imagen dilatada
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar los contornos por área mínima
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]

    # Dibujar cuadros delimitadores alrededor de los contornos
    if kwargs.get('draw_mode', 0) == 0:
        result_image = draw_contours(new_image, large_contours)
    elif kwargs.get('draw_mode', 0) == 1:
        result_image = draw_contours(thresh, large_contours)

    return result_image

# Inicializar la captura de video
cap = cv2.VideoCapture("Practica1/Vídeo 2.mp4")

# Leer el primer fotograma
ret, prev_frame = cap.read()

while cap.isOpened():
    # Leer el fotograma actual
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break

    # Procesar el fotograma actual y mostrarlo
    processed_frame = process_frame_difference_full(frame, prev_frame, min_area_threshold=2000, draw_mode=0)
    cv2.imshow('Processed Frame', processed_frame)
    prev_frame = frame.copy()

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
