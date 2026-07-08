import os
import cv2
import numpy as np

from classificacao import treinar_knn, classificar_superficie_knn


def avaliar_dataset(caminho_dataset):
    knn, media, desvio = treinar_knn(caminho_dataset, k=3)

    total = 0
    acertos = 0

    classes = {
        "normal": "SUPERFICIE NORMAL",
        "defeito": "SUPERFICIE COM DEFEITO"
    }

    for pasta, esperado in classes.items():
        caminho_pasta = os.path.join(caminho_dataset, pasta)

        for arquivo in os.listdir(caminho_pasta):
            if not arquivo.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            caminho = os.path.join(caminho_pasta, arquivo)
            imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

            if imagem is None:
                continue

            obtido, _ = classificar_superficie_knn(imagem, knn, media, desvio, k=3)

            total += 1

            if obtido == esperado:
                acertos += 1

            print(f"{arquivo} | Esperado: {esperado} | Obtido: {obtido}")

    taxa = (acertos / total) * 100 if total > 0 else 0

    print("\n===== AVALIACAO K-NN =====")
    print(f"Total de imagens: {total}")
    print(f"Acertos: {acertos}")
    print(f"Taxa de acerto: {taxa:.2f}%")