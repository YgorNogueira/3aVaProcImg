import cv2

def detectar_bordas(imagem):
    # Detecta bordas usando  Canny.

    bordas = cv2.Canny(imagem, 50, 150)

    return bordas