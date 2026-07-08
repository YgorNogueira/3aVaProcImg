import cv2
import numpy as np


def segmentar_defeitos(imagem):
    # Segmentação com otsu + blackhat

    _, otsu = cv2.threshold(
        imagem,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    abertura = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel, iterations=2)

    return otsu, fechamento