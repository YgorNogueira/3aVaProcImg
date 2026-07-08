# Sistema Inteligente de Inspeção Automática de Superfícies

Projeto desenvolvido para a disciplina de **Processamento Digital de Imagens**, com o objetivo de realizar a inspeção automática de superfícies metálicas utilizando técnicas clássicas de Visão Computacional e o algoritmo de classificação **k-Nearest Neighbors (k-NN)**.

O sistema realiza um pipeline completo de processamento de imagens, desde o pré-processamento até a classificação automática entre **Superfície Normal** e **Superfície com Defeito**, além de gerar automaticamente métricas, gráficos e imagens utilizadas para avaliação do modelo.

---

# Objetivo

Desenvolver um sistema capaz de identificar automaticamente defeitos superficiais utilizando técnicas clássicas de Processamento Digital de Imagens, combinadas com descritores de textura e classificação supervisionada.

---

# Base de Dados

O projeto utiliza uma adaptação da base:

**NEU Surface Defect Database**

Disponível em:

https://github.com/abin24/NEU-Surface-Defect-Database

Para o projeto foi construída uma base binária contendo:

- 20 imagens classificadas como **Superfície Normal**
- 20 imagens classificadas como **Superfície com Defeito**

---

# Pipeline do Sistema

O processamento ocorre na seguinte sequência:

```
Imagem Original
        │
        ▼
Filtro Gaussiano
        │
        ▼
Equalização de Histograma
        │
        ▼
Detector de Bordas (Canny)
        │
        ▼
Transformada Discreta de Fourier (FFT)
        │
        ▼
Segmentação por Otsu
        │
        ▼
Extração de Características (LBP)
        │
        ▼
Classificação (k-NN)
        │
        ▼
Resultado Final
```

---

# Técnicas Utilizadas

## Pré-processamento

- Filtro Gaussiano
- Equalização de Histograma

## Detecção de Bordas

- Canny Edge Detector

## Análise no Domínio da Frequência

- Transformada Discreta de Fourier (FFT)

## Segmentação

- Método de Otsu
- Operações Morfológicas

## Extração de Características

- Local Binary Pattern (LBP)

## Classificação

- k-Nearest Neighbors (k-NN)

---

# Tecnologias Utilizadas

- Python 3.12+
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Scikit-image
- Scikit-learn

---

# Estrutura do Projeto

```
Projeto/

│
├── dataset/
│   ├── normal/
│   └── defeito/
│
├── resultados/
│   ├── pipeline.png
│   ├── metricas.png
│   ├── matriz_confusao.png
│   ├── resultados.csv
│   ├── metricas.txt
│   └── amostras/
│
├── src/
│   ├── preprocessamento.py
│   ├── bordas.py
│   ├── frequencia.py
│   ├── segmentacao.py
│   ├── classificacao.py
│   ├── avaliacao.py
│   ├── gerar_resultados.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

# Como Executar

## 1. Clone o repositório

```bash
git clone <url-do-repositorio>

cd Projeto
```

---

## 2. Crie um ambiente virtual (Opcional)

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 4. Organize o Dataset

```
dataset/

├── normal/

└── defeito/
```

---

## 5. Execute o sistema

Entre na pasta **src**

```bash
cd src
```

Execute:

```bash
python main.py
```

---

# Saídas Geradas

Após a execução serão gerados automaticamente:

```
resultados/

pipeline.png

metricas.png

matriz_confusao.png

metricas.txt

resultados.csv

amostras/
```

Além disso, durante a execução o sistema apresenta:

- Aplicação do filtro Gaussiano
- Equalização
- Detector de bordas
- FFT
- Segmentação
- Demonstração visual do pipeline
- Avaliação completa do modelo

---

# Métricas Avaliadas

O sistema calcula automaticamente:

- Acurácia
- Precisão
- Recall
- F1-score
- Matriz de Confusão

---

# Exemplo de Resultado

| Métrica | Valor |
|----------|-------|
| Acurácia | 92,50% |
| Precisão | 90,48% |
| Recall | 95,00% |
| F1-score | 92,68% |

---

# Autor

**Ygor Fellipe Macêdo Nogueira**

Universidade Federal Rural de Pernambuco

Disciplina: Processamento de Imagens

Professor: Lucas Cambuim

---

# Licença

Projeto desenvolvido exclusivamente para fins acadêmicos.