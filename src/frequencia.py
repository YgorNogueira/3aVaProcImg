import cv2
import numpy as np


def transformar_fft(imagem):
    # Calcula a Transformada Discreta de Fourier e retorna o do espectro de frequência em uma img

    fft = np.fft.fft2(imagem)

    fft_shift = np.fft.fftshift(fft)

    magnitude = 20 * np.log(np.abs(fft_shift) + 1)


    # Normaliza pra exibir
    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude.astype(np.uint8)