import os
import cv2

from preprocessamento import preprocessar
from bordas import detectar_bordas
from frequencia import transformar_fft
from segmentacao import segmentar_defeitos
from classificacao import treinar_knn, classificar_superficie_knn


def processar_imagem(caminho_imagem, caminho_dataset="../dataset"):
    knn, media, desvio = treinar_knn(caminho_dataset, k=3)

    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        print("Erro ao carregar imagem.")
        return

    gauss, equalizada = preprocessar(imagem)
    bordas = detectar_bordas(equalizada)
    fft = transformar_fft(equalizada)
    otsu, segmentada = segmentar_defeitos(equalizada)

    resultado, _ = classificar_superficie_knn(imagem, knn, media, desvio, k=3)

    print("\n===== RESULTADO DA INSPEÇÃO =====")
    print(f"Imagem analisada: {caminho_imagem}")
    print(f"Classificação: {resultado}")

    cv2.imshow("Original", imagem)
    cv2.imshow("Filtro Gaussiano", gauss)
    cv2.imshow("Equalizacao de Histograma", equalizada)
    cv2.imshow("Bordas - Canny", bordas)
    cv2.imshow("Dominio da Frequencia - FFT", fft)
    cv2.imshow("Segmentacao - Otsu", otsu)
    cv2.imshow("Segmentacao Final", segmentada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    caminho = "../dataset/defeito/crazing_285.jpg"
    processar_imagem(caminho)