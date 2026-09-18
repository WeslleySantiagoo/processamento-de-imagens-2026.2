"""Q2 — Ajuste de brilho por correção gama."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "baboon_monocromatica.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q2_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega a imagem como uma matriz de intensidades inteiras entre 0 e 255."""
    imagem = plt.imread(caminho)

    if np.issubdtype(imagem.dtype, np.floating):
        imagem = np.rint(imagem * 255).astype(np.uint8)

    if imagem.ndim == 3:
        imagem = imagem[..., 0]

    return imagem


def correcao_gama(imagem: np.ndarray, gama: float):
    """Aplica B = A^(1/gama), com A e B no intervalo de intensidades [0, 255]."""
    if gama <= 0:
        raise ValueError("O valor de gama deve ser maior que zero.")

    intensidade_maxima = 255

    imagem_normalizada = imagem.astype(float) / intensidade_maxima

    expoente = 1 / gama
    imagem_corrigida = imagem_normalizada ** expoente

    imagem_255 = imagem_corrigida * intensidade_maxima
    return np.rint(imagem_255).astype(np.uint8)


def salvar_comparacao(original: np.ndarray, imagem_gama_1: np.ndarray,
                      imagem_gama_2: np.ndarray, imagem_gama_3: np.ndarray,
                      gama_1: float, gama_2: float, gama_3: float, saida: Path):
    """Exibe e salva a imagem original e três versões com correção gama."""
    # Usado IA apenas para amostragem visual
    figura, eixos = plt.subplots(1, 4, figsize=(14, 4))
    figura.subplots_adjust(wspace=0.08, bottom=0.14)

    e0, e1, e2, e3 = eixos

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(imagem_gama_1, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(imagem_gama_2, cmap="gray", vmin=0, vmax=255); e2.axis("off")
    e3.imshow(imagem_gama_3, cmap="gray", vmin=0, vmax=255); e3.axis("off")

    legenda_gama_1 = f"(b) γ = {gama_1}".replace(".", ",")
    legenda_gama_2 = f"(c) γ = {gama_2}".replace(".", ",")
    legenda_gama_3 = f"(d) γ = {gama_3}".replace(".", ",")

    e0.text(0.5, -0.08, "(a) imagem original", transform=e0.transAxes,
            ha="center", va="top", fontsize=11)
    e1.text(0.5, -0.08, legenda_gama_1, transform=e1.transAxes,
            ha="center", va="top", fontsize=11)
    e2.text(0.5, -0.08, legenda_gama_2, transform=e2.transAxes,
            ha="center", va="top", fontsize=11)
    e3.text(0.5, -0.08, legenda_gama_3, transform=e3.transAxes,
            ha="center", va="top", fontsize=11)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)

    gama_1 = 1.5
    gama_2 = 2.5
    gama_3 = 3.5

    resultado_gama_1 = correcao_gama(original, gama_1)
    resultado_gama_2 = correcao_gama(original, gama_2)
    resultado_gama_3 = correcao_gama(original, gama_3)

    salvar_comparacao(original, resultado_gama_1, resultado_gama_2, resultado_gama_3,
                      gama_1, gama_2, gama_3, ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
