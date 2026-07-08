import os
import cv2
import numpy as np

from preprocessamento import preprocessar
from bordas import detectar_bordas
from frequencia import transformar_fft
from segmentacao import segmentar_defeitos
from classificacao import treinar_knn, classificar_superficie_knn


def colocar_titulo(imagem, titulo):
    imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        imagem,
        titulo,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )
    return imagem


def redimensionar(imagem, tamanho=(250, 250)):
    return cv2.resize(imagem, tamanho)


def gerar_tabela_visual(caminho_imagem, caminho_dataset):
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

    imagem_resultado = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

    cor = (0, 0, 255) if "DEFEITO" in resultado else (0, 255, 0)

    cv2.putText(
        imagem_resultado,
        resultado,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        cor,
        2
    )

    img1 = colocar_titulo(redimensionar(imagem), "Original")
    img2 = colocar_titulo(redimensionar(gauss), "Filtro Gaussiano")
    img3 = colocar_titulo(redimensionar(equalizada), "Equalizacao")
    img4 = colocar_titulo(redimensionar(bordas), "Bordas - Canny")
    img5 = colocar_titulo(redimensionar(fft), "FFT")
    img6 = colocar_titulo(redimensionar(segmentada), "Segmentacao")
    img7 = redimensionar(imagem_resultado)

    linha1 = np.hstack([img1, img2, img3])
    linha2 = np.hstack([img4, img5, img6])

    vazio = np.zeros_like(img1)
    linha3 = np.hstack([vazio, img7, vazio])

    tabela = np.vstack([linha1, linha2, linha3])

    os.makedirs("../resultados", exist_ok=True)

    caminho_saida = "../resultados/tabela_resultado.png"
    cv2.imwrite(caminho_saida, tabela)

    cv2.imshow("Pipeline completo do sistema", tabela)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Imagem salva em: {caminho_saida}")


gerar_tabela_visual(
    "../dataset/defeito/crazing_285.jpg",
    "../dataset"
)