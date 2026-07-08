import os
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from classificacao import treinar_knn, classificar_superficie_knn


def avaliar_dataset(caminho_dataset):
    knn, scaler = treinar_knn(caminho_dataset, k=3)

    y_real = []
    y_pred = []

    classes = {
        "normal": 0,
        "defeito": 1
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

            resultado, confianca = classificar_superficie_knn(imagem, knn, scaler)

            previsto = 1 if resultado == "SUPERFICIE COM DEFEITO" else 0

            y_real.append(esperado)
            y_pred.append(previsto)

            print(f"{arquivo} | Esperado: {esperado} | Previsto: {previsto} | Confiança: {confianca:.2f}%")

    acc = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred, zero_division=0)
    rec = recall_score(y_real, y_pred, zero_division=0)
    f1 = f1_score(y_real, y_pred, zero_division=0)
    matriz = confusion_matrix(y_real, y_pred)

    print("\n===== AVALIACAO K-NN COM LBP =====")
    print(f"Acurácia: {acc * 100:.2f}%")
    print(f"Precisão: {prec * 100:.2f}%")
    print(f"Recall: {rec * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")
    print("\nMatriz de confusão:")
    print(matriz)