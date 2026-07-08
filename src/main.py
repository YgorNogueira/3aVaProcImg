import os
import cv2

from preprocessamento import preprocessar
from bordas import detectar_bordas
from frequencia import transformar_fft
from segmentacao import segmentar_defeitos
from classificacao import treinar_knn, classificar_superficie_knn

from gerar_resultados import (
    garantir_pastas,
    avaliar_e_salvar_resultados,
    salvar_pipeline,
    gerar_amostras,
    CAMINHO_DATASET
)


def processar_imagem(caminho_imagem, knn, scaler):
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        print("Erro ao carregar imagem.")
        return

    gauss, equalizada = preprocessar(imagem)
    bordas = detectar_bordas(equalizada)
    fft = transformar_fft(equalizada)
    otsu, segmentada = segmentar_defeitos(equalizada)

    resultado, confianca = classificar_superficie_knn(imagem, knn, scaler)

    print("\n===== RESULTADO DA INSPEÇÃO =====")
    print(f"Imagem analisada: {caminho_imagem}")
    print(f"Classificação: {resultado}")
    print(f"Confiança: {confianca:.2f}%")

    cv2.imshow("Original", imagem)
    cv2.imshow("Filtro Gaussiano", gauss)
    cv2.imshow("Equalizacao de Histograma", equalizada)
    cv2.imshow("Bordas - Canny", bordas)
    cv2.imshow("Dominio da Frequencia - FFT", fft)
    cv2.imshow("Segmentacao - Otsu", otsu)
    cv2.imshow("Segmentacao Final", segmentada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def escolher_imagem_exemplo():
    pasta_defeito = os.path.join(CAMINHO_DATASET, "defeito")

    arquivos = [
        a for a in os.listdir(pasta_defeito)
        if a.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not arquivos:
        return None

    return os.path.join(pasta_defeito, arquivos[0])


def mostrar_demo_visual():
    caminho_pipeline = "../resultados/pipeline.png"

    imagem = cv2.imread(caminho_pipeline)

    if imagem is None:
        print("Não foi possível abrir a demo visual.")
        return

    largura = 1200
    altura = int(imagem.shape[0] * (largura / imagem.shape[1]))
    imagem = cv2.resize(imagem, (largura, altura))

    cv2.imshow("Demo Visual - Pipeline Completo", imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("=" * 70)
    print("SISTEMA INTELIGENTE DE INSPEÇÃO AUTOMÁTICA DE SUPERFÍCIES")
    print("Processamento Digital de Imagens + k-NN")
    print("=" * 70)

    garantir_pastas()

    print("\n[1/5] Treinando classificador k-NN...")
    knn, scaler = treinar_knn(CAMINHO_DATASET, k=3)

    print("\n[2/5] Executando técnicas de processamento em uma imagem exemplo...")
    imagem_exemplo = escolher_imagem_exemplo()

    if imagem_exemplo is None:
        print("Nenhuma imagem encontrada em dataset/defeito.")
        return

    processar_imagem(imagem_exemplo, knn, scaler)

    print("\n[3/5] Gerando demo visual do pipeline completo...")
    salvar_pipeline(imagem_exemplo, knn, scaler, "pipeline.png")
    mostrar_demo_visual()

    print("\n[4/5] Avaliando o dataset completo...")
    df_resultados = avaliar_e_salvar_resultados(knn, scaler)

    print("\n[5/5] Gerando amostras visuais para relatório/slides...")
    gerar_amostras(df_resultados, knn, scaler, limite=6)

    print("\n" + "=" * 70)
    print("EXECUÇÃO FINALIZADA COM SUCESSO!")
    print("Arquivos gerados em: resultados/")
    print("=" * 70)

    print("""
Arquivos principais:

- resultados/pipeline.png
- resultados/metricas.png
- resultados/matriz_confusao.png
- resultados/metricas.txt
- resultados/resultados.csv
- resultados/amostras/
""")


if __name__ == "__main__":
    main()