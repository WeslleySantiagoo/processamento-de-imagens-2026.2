"""Q2 — Ampliação de imagem pela técnica de vizinho mais próximo."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "baboon_monocromatica.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q2_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega uma imagem PNG como matriz bidimensional de intensidades."""
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")

    return imagem


def ampliar_por_vizinho_mais_proximo(imagem: np.ndarray, fator: int):
    """Amplia a imagem pelo fator informado usando índices do vizinho mais próximo."""
    if fator <= 0:
        raise ValueError("O fator de ampliação deve ser maior que zero.")

    altura_original, largura_original = imagem.shape
    altura_ampliada = altura_original * fator
    largura_ampliada = largura_original * fator

    linhas_saida = np.arange(altura_ampliada)
    escala_vertical = altura_original / altura_ampliada
    linhas_origem = np.rint(linhas_saida * escala_vertical).astype(int)
    linhas_origem = np.clip(linhas_origem, 0, altura_original - 1)

    colunas_saida = np.arange(largura_ampliada)
    escala_horizontal = largura_original / largura_ampliada
    colunas_origem = np.rint(colunas_saida * escala_horizontal).astype(int)
    colunas_origem = np.clip(colunas_origem, 0, largura_original - 1)

    imagem_ampliada = imagem[linhas_origem[:, np.newaxis], colunas_origem]

    return imagem_ampliada


def salvar_comparacao(original: np.ndarray, ampliada_2: np.ndarray,
                      ampliada_4: np.ndarray, saida: Path):
    """Exibe e salva a imagem original e suas ampliações de 2x e 4x."""
    # Usado IA apenas para amostragem visual
    saida.parent.mkdir(parents=True, exist_ok=True)

    figura = plt.figure(figsize=(12, 5))
    grade = figura.add_gridspec(1, 5, width_ratios=[1, 0.35, 2, 0.10, 4],
                                wspace=0.0)

    e0 = figura.add_subplot(grade[0, 0])
    e1 = figura.add_subplot(grade[0, 2])
    e2 = figura.add_subplot(grade[0, 4])

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(ampliada_2, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(ampliada_4, cmap="gray", vmin=0, vmax=255); e2.axis("off")

    e0.text(0.5, -0.08, "(a) imagem original", transform=e0.transAxes,
            ha="center", va="top", fontsize=11)
    e1.text(0.5, -0.08, "(b) ampliação por 2", transform=e1.transAxes,
            ha="center", va="top", fontsize=11)
    e2.text(0.5, -0.08, "(c) ampliação por 4", transform=e2.transAxes,
            ha="center", va="top", fontsize=11)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    fator_1 = 2
    fator_2 = 4

    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)
    ampliada_2 = ampliar_por_vizinho_mais_proximo(original, fator_1)
    ampliada_4 = ampliar_por_vizinho_mais_proximo(original, fator_2)

    salvar_comparacao(original, ampliada_2, ampliada_4, ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
