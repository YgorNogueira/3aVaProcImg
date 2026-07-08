import os
import cv2
import itertools

from preprocessamento import preprocessar
from bordas import detectar_bordas
from segmentacao import segmentar_defeitos
from classificacao import extrair_caracteristicas


def carregar_caracteristicas(caminho_dataset):
    dados = []

    classes = {
        "positivo": 1,
        "negativo": 0
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

            _, equalizada = preprocessar(imagem)
            bordas = detectar_bordas(equalizada)
            _, segmentada = segmentar_defeitos(equalizada)

            carac = extrair_caracteristicas(equalizada, segmentada, bordas)

            dados.append({
                "arquivo": arquivo,
                "esperado": esperado,
                "regioes_lineares": carac["regioes_lineares"],
                "densidade_bordas": carac["densidade_bordas"],
                "percentual_area": carac["percentual_area"],
                "quantidade_regioes": carac["quantidade_regioes"]
            })

    return dados


def testar_parametros(dados):
    melhor = {
        "acertos": 0,
        "taxa": 0,
        "params": None
    }

    limites_regioes_lineares = [0, 1, 2, 3, 4, 5]
    limites_densidade = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
    limites_area_min = [0.001, 0.005, 0.01, 0.02]
    limites_area_max = [0.15, 0.25, 0.35, 0.50, 0.70]
    limites_quantidade = [1, 3, 5, 8, 10]
    limites_score = [1, 2, 3, 4]

    combinacoes = itertools.product(
        limites_regioes_lineares,
        limites_densidade,
        limites_area_min,
        limites_area_max,
        limites_quantidade,
        limites_score
    )

    for reg_lin, dens, area_min, area_max, qtd, score_min in combinacoes:
        acertos = 0

        for item in dados:
            score = 0

            if item["regioes_lineares"] >= reg_lin:
                score += 2

            if item["densidade_bordas"] >= dens:
                score += 2

            if area_min <= item["percentual_area"] <= area_max:
                score += 1

            if item["quantidade_regioes"] >= qtd:
                score += 1

            previsto = 1 if score >= score_min else 0

            if previsto == item["esperado"]:
                acertos += 1

        taxa = acertos / len(dados) * 100

        if taxa > melhor["taxa"]:
            melhor = {
                "acertos": acertos,
                "taxa": taxa,
                "params": {
                    "regioes_lineares": reg_lin,
                    "densidade_bordas": dens,
                    "area_min": area_min,
                    "area_max": area_max,
                    "quantidade_regioes": qtd,
                    "score_min": score_min
                }
            }

    return melhor


dados = carregar_caracteristicas("../dataset")
melhor = testar_parametros(dados)

print("\n===== MELHOR CONFIGURAÇÃO ENCONTRADA =====")
print(f"Acertos: {melhor['acertos']} de {len(dados)}")
print(f"Taxa: {melhor['taxa']:.2f}%")
print("Parâmetros:")
for chave, valor in melhor["params"].items():
    print(f"{chave}: {valor}")