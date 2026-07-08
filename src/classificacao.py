import os
import cv2
import numpy as np

from preprocessamento import preprocessar
from bordas import detectar_bordas
from frequencia import transformar_fft
from segmentacao import segmentar_defeitos


def extrair_caracteristicas(imagem):
    _, equalizada = preprocessar(imagem)
    bordas = detectar_bordas(equalizada)
    fft = transformar_fft(equalizada)
    _, segmentada = segmentar_defeitos(equalizada)

    altura, largura = imagem.shape
    area_total = altura * largura

    contornos, _ = cv2.findContours(
        segmentada,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    areas = []
    regioes_lineares = 0

    for c in contornos:
        area = cv2.contourArea(c)

        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(c)
        proporcao = max(w, h) / (min(w, h) + 1)

        areas.append(area)

        if proporcao >= 3:
            regioes_lineares += 1

    area_segmentada = sum(areas)
    percentual_area = area_segmentada / area_total
    densidade_bordas = np.count_nonzero(bordas) / area_total

    media_intensidade = np.mean(equalizada)
    desvio_intensidade = np.std(equalizada)
    media_fft = np.mean(fft)
    desvio_fft = np.std(fft)

    return np.array([
        percentual_area,
        densidade_bordas,
        regioes_lineares,
        len(areas),
        media_intensidade,
        desvio_intensidade,
        media_fft,
        desvio_fft
    ], dtype=np.float32)


def carregar_dataset(caminho_dataset):
    X = []
    y = []

    classes = {
        "normal": 0,
        "defeito": 1
    }

    for pasta, label in classes.items():
        caminho_pasta = os.path.join(caminho_dataset, pasta)

        for arquivo in os.listdir(caminho_pasta):
            if not arquivo.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            caminho = os.path.join(caminho_pasta, arquivo)
            imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

            if imagem is None:
                continue

            caracteristicas = extrair_caracteristicas(imagem)

            X.append(caracteristicas)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def normalizar_dados(X):
    media = np.mean(X, axis=0)
    desvio = np.std(X, axis=0)
    desvio[desvio == 0] = 1

    X_norm = (X - media) / desvio

    return X_norm, media, desvio


def treinar_knn(caminho_dataset, k=3):
    X, y = carregar_dataset(caminho_dataset)

    X_norm, media, desvio = normalizar_dados(X)

    knn = cv2.ml.KNearest_create()
    knn.train(X_norm, cv2.ml.ROW_SAMPLE, y)

    return knn, media, desvio


def classificar_superficie_knn(imagem, knn, media, desvio, k=3):
    caracteristicas = extrair_caracteristicas(imagem)
    caracteristicas_norm = (caracteristicas - media) / desvio
    caracteristicas_norm = np.array([caracteristicas_norm], dtype=np.float32)

    _, resultado, _, distancias = knn.findNearest(caracteristicas_norm, k)

    classe = int(resultado[0][0])

    if classe == 1:
        texto = "SUPERFICIE COM DEFEITO"
    else:
        texto = "SUPERFICIE NORMAL"

    return texto, distancias