"""Q1 — Visualização da Transformada de Fourier Discreta 2D."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "city.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q1_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega uma imagem como matriz de intensidades em tons de cinza."""
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")

    return imagem


def calcular_componentes_fourier(imagem: np.ndarray):
    """Calcula espectro centralizado, magnitude logarítmica e fase normalizada."""
    imagem_float = imagem.astype(np.float32)
    espectro = np.fft.fft2(imagem_float)
    espectro_centralizado = np.fft.fftshift(espectro)

    magnitude = 20 * np.log(np.abs(espectro_centralizado) + 1)
    fase = np.angle(espectro_centralizado)
    fase_normalizada = (fase + np.pi) / (2 * np.pi)

    return espectro_centralizado, magnitude, fase_normalizada


def salvar_comparacao(original: np.ndarray, magnitude: np.ndarray,
                      fase: np.ndarray, saida: Path):
    """Exibe e salva a imagem original, a magnitude e a fase do espectro."""
    # Usado IA para melhor amosytragem
    saida.parent.mkdir(parents=True, exist_ok=True)

    figura, eixos = plt.subplots(1, 3, figsize=(13, 5))
    figura.subplots_adjust(wspace=0.05, bottom=0.15)
    e0, e1, e2 = eixos

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(magnitude, cmap="gray"); e1.axis("off")
    e2.imshow(fase, cmap="gray", vmin=0, vmax=1); e2.axis("off")

    e0.text(0.5, -0.08, "(a) imagem original", transform=e0.transAxes,
            ha="center", va="top", fontsize=11)
    e1.text(0.5, -0.08, "(b) espectro de magnitude", transform=e1.transAxes,
            ha="center", va="top", fontsize=11)
    e2.text(0.5, -0.08, "(c) espectro de fase", transform=e2.transAxes,
            ha="center", va="top", fontsize=11)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)
    _, magnitude, fase_normalizada = calcular_componentes_fourier(original)

    salvar_comparacao(original, magnitude, fase_normalizada, ARQUIVO_SAIDA)
    print(f"Imagem usada: {ARQUIVO_ENTRADA.name}")
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
