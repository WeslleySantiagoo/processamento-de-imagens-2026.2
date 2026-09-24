"""Q1 — Rotação de imagem em múltiplos de 90 graus."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "baboon_monocromatica.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q1_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega uma imagem PNG como matriz bidimensional de intensidades."""
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")

    return imagem


def rotacionar_90_horario(imagem: np.ndarray):
    """Rotaciona a matriz 90 graus no sentido horário por mapeamento de índices."""
    altura, largura = imagem.shape
    linhas, colunas = np.indices(imagem.shape)
    rotacionada = np.empty((largura, altura), dtype=imagem.dtype)

    rotacionada[colunas, altura - linhas - 1] = imagem

    return rotacionada


def rotacionar_180(imagem: np.ndarray):
    """Rotaciona a matriz 180 graus por mapeamento de índices."""
    altura, largura = imagem.shape
    linhas, colunas = np.indices(imagem.shape)
    rotacionada = np.empty_like(imagem)

    rotacionada[altura - linhas - 1, largura - colunas - 1] = imagem

    return rotacionada


def rotacionar_270_horario(imagem: np.ndarray):
    """Rotaciona a matriz 270 graus no sentido horário por mapeamento de índices."""
    altura, largura = imagem.shape
    linhas, colunas = np.indices(imagem.shape)
    rotacionada = np.empty((largura, altura), dtype=imagem.dtype)

    rotacionada[largura - colunas - 1, linhas] = imagem

    return rotacionada


def salvar_comparacao(original: np.ndarray, rotacao_90: np.ndarray,
                      rotacao_180: np.ndarray, rotacao_270: np.ndarray,
                      saida: Path):
    """Exibe e salva a comparação entre a imagem original e as rotações."""
    # Usado IA apenas para amostragem visual
    saida.parent.mkdir(parents=True, exist_ok=True)

    figura, eixos = plt.subplots(2, 2, figsize=(9, 9))
    figura.subplots_adjust(wspace=0.06, hspace=0.20, bottom=0.08)
    e0, e1, e2, e3 = eixos.flat

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(rotacao_90, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(rotacao_180, cmap="gray", vmin=0, vmax=255); e2.axis("off")
    e3.imshow(rotacao_270, cmap="gray", vmin=0, vmax=255); e3.axis("off")

    e0.text(0.5, -0.08, "(a) imagem original", transform=e0.transAxes,
            ha="center", va="top", fontsize=12)
    e1.text(0.5, -0.08, "(b) rotação 90° horário", transform=e1.transAxes,
            ha="center", va="top", fontsize=12)
    e2.text(0.5, -0.08, "(c) rotação 180°", transform=e2.transAxes,
            ha="center", va="top", fontsize=12)
    e3.text(0.5, -0.08, "(d) rotação 270° horário", transform=e3.transAxes,
            ha="center", va="top", fontsize=12)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)

    rotacao_90 = rotacionar_90_horario(original)
    rotacao_180 = rotacionar_180(original)
    rotacao_270 = rotacionar_270_horario(original)

    salvar_comparacao(original, rotacao_90, rotacao_180, rotacao_270, ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
