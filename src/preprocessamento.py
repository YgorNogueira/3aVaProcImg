import cv2

def preprocessar(imagem):
    """
    Realiza o pré-processamento da imagem.

    Etapas:
    1. Filtro Gaussiano
    2. Equalização de Histograma
    """

    # Redução de ruído
    gauss = cv2.GaussianBlur(imagem, (5, 5), 0)

    # Melhora do contraste
    equalizada = cv2.equalizeHist(gauss)

    return gauss, equalizada