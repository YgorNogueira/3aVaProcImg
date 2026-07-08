import os
import cv2
import numpy as np

from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from preprocessamento import preprocessar


def extrair_lbp(imagem):
    _, equalizada = preprocessar(imagem)

    radius = 2
    n_points = 8 * radius

    lbp = local_binary_pattern(
        equalizada,
        n_points,
        radius,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return hist


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
            if arquivo.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                caminho = os.path.join(caminho_pasta, arquivo)

                imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

                if imagem is None:
                    continue

                caracteristicas = extrair_lbp(imagem)

                X.append(caracteristicas)
                y.append(label)

    return np.array(X), np.array(y)


def treinar_knn(caminho_dataset, k=3):
    X, y = carregar_dataset(caminho_dataset)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_norm, y)

    return knn, scaler


def classificar_superficie_knn(imagem, knn, scaler):
    caracteristicas = extrair_lbp(imagem)
    caracteristicas = scaler.transform([caracteristicas])

    classe = knn.predict(caracteristicas)[0]
    probabilidades = knn.predict_proba(caracteristicas)[0]

    confianca = max(probabilidades) * 100

    if classe == 1:
        resultado = "SUPERFICIE COM DEFEITO"
    else:
        resultado = "SUPERFICIE NORMAL"

    return resultado, confianca