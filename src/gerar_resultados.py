import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessamento import preprocessar
from bordas import detectar_bordas
from frequencia import transformar_fft
from segmentacao import segmentar_defeitos
from classificacao import treinar_knn, classificar_superficie_knn, extrair_lbp


CAMINHO_DATASET = "../dataset"
CAMINHO_RESULTADOS = "../resultados"


def garantir_pastas():
    os.makedirs(CAMINHO_RESULTADOS, exist_ok=True)
    os.makedirs(os.path.join(CAMINHO_RESULTADOS, "amostras"), exist_ok=True)


def salvar_pipeline(caminho_imagem, knn, scaler, nome_saida="pipeline.png"):
    inicio = time.time()

    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        print(f"Erro ao carregar imagem: {caminho_imagem}")
        return

    gauss, equalizada = preprocessar(imagem)
    bordas = detectar_bordas(equalizada)
    fft = transformar_fft(equalizada)
    otsu, segmentada = segmentar_defeitos(equalizada)
    lbp = extrair_lbp(imagem)

    resultado, confianca = classificar_superficie_knn(imagem, knn, scaler)

    tempo = time.time() - inicio

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    imagens = [
        (imagem, "Original"),
        (gauss, "Filtro Gaussiano"),
        (equalizada, "Equalização"),
        (bordas, "Bordas - Canny"),
        (fft, "FFT"),
        (otsu, "Otsu"),
        (segmentada, "Segmentação"),
        (imagem, resultado)
    ]

    for ax, (img, titulo) in zip(axs.ravel(), imagens):
        ax.imshow(img, cmap="gray")
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"Sistema de Inspeção Automática de Superfícies\n"
        f"Classe: {resultado} | Confiança: {confianca:.2f}% | Tempo: {tempo:.4f}s",
        fontsize=16
    )

    plt.tight_layout()
    caminho_saida = os.path.join(CAMINHO_RESULTADOS, nome_saida)
    plt.savefig(caminho_saida, dpi=300)
    plt.close()

    print(f"Pipeline salvo em: {caminho_saida}")


def avaliar_e_salvar_resultados(knn, scaler):
    y_real = []
    y_pred = []
    registros = []

    classes = {
        "normal": 0,
        "defeito": 1
    }

    for pasta, esperado in classes.items():
        caminho_pasta = os.path.join(CAMINHO_DATASET, pasta)

        for arquivo in os.listdir(caminho_pasta):
            if not arquivo.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            caminho_img = os.path.join(caminho_pasta, arquivo)

            inicio = time.time()
            imagem = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)

            if imagem is None:
                continue

            resultado, confianca = classificar_superficie_knn(imagem, knn, scaler)
            tempo = time.time() - inicio

            previsto = 1 if resultado == "SUPERFICIE COM DEFEITO" else 0
            acertou = esperado == previsto

            y_real.append(esperado)
            y_pred.append(previsto)

            registros.append({
                "imagem": arquivo,
                "classe_real": "SUPERFICIE COM DEFEITO" if esperado == 1 else "SUPERFICIE NORMAL",
                "classe_prevista": resultado,
                "confianca": round(confianca, 2),
                "tempo_processamento": round(tempo, 5),
                "acertou": acertou
            })

    acc = accuracy_score(y_real, y_pred)
    prec = precision_score(y_real, y_pred, zero_division=0)
    rec = recall_score(y_real, y_pred, zero_division=0)
    f1 = f1_score(y_real, y_pred, zero_division=0)
    matriz = confusion_matrix(y_real, y_pred)

    df = pd.DataFrame(registros)
    df.to_csv(os.path.join(CAMINHO_RESULTADOS, "resultados.csv"), index=False, encoding="utf-8-sig")

    salvar_metricas_txt(acc, prec, rec, f1, matriz, len(y_real))
    salvar_grafico_metricas(acc, prec, rec, f1)
    salvar_matriz_confusao(matriz)

    print("\n===== RESULTADOS =====")
    print(f"Acurácia: {acc * 100:.2f}%")
    print(f"Precisão: {prec * 100:.2f}%")
    print(f"Recall: {rec * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")
    print("\nMatriz de confusão:")
    print(matriz)

    return df


def salvar_metricas_txt(acc, prec, rec, f1, matriz, total):
    caminho = os.path.join(CAMINHO_RESULTADOS, "metricas.txt")

    with open(caminho, "w", encoding="utf-8") as f:
        f.write("Sistema Inteligente de Inspeção Automática de Superfícies\n")
        f.write("Utilizando Processamento Digital de Imagens e k-NN\n\n")
        f.write("Base de dados: NEU Surface Defect Database\n")
        f.write(f"Total de imagens avaliadas: {total}\n\n")
        f.write(f"Acurácia: {acc * 100:.2f}%\n")
        f.write(f"Precisão: {prec * 100:.2f}%\n")
        f.write(f"Recall: {rec * 100:.2f}%\n")
        f.write(f"F1-score: {f1 * 100:.2f}%\n\n")
        f.write("Matriz de confusão:\n")
        f.write(str(matriz))

    print(f"Métricas salvas em: {caminho}")


def salvar_grafico_metricas(acc, prec, rec, f1):
    nomes = ["Acurácia", "Precisão", "Recall", "F1-score"]
    valores = [acc * 100, prec * 100, rec * 100, f1 * 100]

    plt.figure(figsize=(8, 5))
    plt.bar(nomes, valores)

    for i, valor in enumerate(valores):
        plt.text(i, valor + 1, f"{valor:.2f}%", ha="center")

    plt.ylim(0, 110)
    plt.title("Métricas de Avaliação do Sistema")
    plt.ylabel("Percentual (%)")
    plt.tight_layout()

    caminho = os.path.join(CAMINHO_RESULTADOS, "metricas.png")
    plt.savefig(caminho, dpi=300)
    plt.close()

    print(f"Gráfico de métricas salvo em: {caminho}")


def salvar_matriz_confusao(matriz):
    plt.figure(figsize=(6, 5))
    plt.imshow(matriz)

    plt.title("Matriz de Confusão")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")

    classes = ["Normal", "Com Defeito"]
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            plt.text(j, i, matriz[i, j], ha="center", va="center", fontsize=16)

    plt.colorbar()
    plt.tight_layout()

    caminho = os.path.join(CAMINHO_RESULTADOS, "matriz_confusao.png")
    plt.savefig(caminho, dpi=300)
    plt.close()

    print(f"Matriz de confusão salva em: {caminho}")


def gerar_amostras(df, knn, scaler, limite=6):
    amostras = df.head(limite)

    for i, linha in enumerate(amostras.itertuples(), start=1):
        classe_pasta = "defeito" if linha.classe_real == "SUPERFICIE COM DEFEITO" else "normal"
        caminho_img = os.path.join(CAMINHO_DATASET, classe_pasta, linha.imagem)

        salvar_pipeline(
            caminho_img,
            knn,
            scaler,
            nome_saida=f"amostras/resultado_{i}.png"
        )


if __name__ == "__main__":
    garantir_pastas()

    print("Treinando k-NN...")
    knn, scaler = treinar_knn(CAMINHO_DATASET, k=3)

    print("Avaliando dataset...")
    df_resultados = avaliar_e_salvar_resultados(knn, scaler)

    exemplo = os.path.join(CAMINHO_DATASET, "defeito", "crazing_285.jpg")

    if os.path.exists(exemplo):
        salvar_pipeline(exemplo, knn, scaler, "pipeline.png")
    else:
        primeira_defeito = os.listdir(os.path.join(CAMINHO_DATASET, "defeito"))[0]
        exemplo = os.path.join(CAMINHO_DATASET, "defeito", primeira_defeito)
        salvar_pipeline(exemplo, knn, scaler, "pipeline.png")

    gerar_amostras(df_resultados, knn, scaler, limite=6)

    print("\nTudo gerado em ../resultados")