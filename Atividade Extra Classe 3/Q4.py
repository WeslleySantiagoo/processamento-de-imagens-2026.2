"""Q4 — Efeito de esboço a lápis por desfoque gaussiano e divisão."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "gray_watch.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q4_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega uma imagem PNG como matriz bidimensional de intensidades."""
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")

    return imagem


def criar_kernel_gaussiano(tamanho: int, sigma: float):
    """Cria um kernel gaussiano quadrado normalizado pela fórmula matemática."""
    if tamanho <= 0 or tamanho % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser um número ímpar positivo.")

    if sigma <= 0:
        raise ValueError("O valor de sigma deve ser maior que zero.")

    centro = tamanho // 2
    coordenadas = np.arange(-centro, centro + 1)
    coordenadas_x, coordenadas_y = np.meshgrid(coordenadas, coordenadas)

    expoente = -(coordenadas_x ** 2 + coordenadas_y ** 2) / (2 * sigma ** 2)
    kernel = np.exp(expoente)

    return kernel / kernel.sum()


def correlacionar_gaussiano_separavel(imagem: np.ndarray, kernel: np.ndarray):
    """Aplica o kernel gaussiano em duas correlações 1D equivalentes à máscara 2D."""
    tamanho = kernel.shape[0]

    if kernel.shape != (tamanho, tamanho) or tamanho % 2 == 0:
        raise ValueError("O kernel gaussiano deve ser quadrado e ter tamanho ímpar.")

    margem = tamanho // 2

    kernel_horizontal = kernel[margem, :]
    kernel_horizontal = kernel_horizontal / kernel_horizontal.sum()
    kernel_vertical = kernel[:, margem]
    kernel_vertical = kernel_vertical / kernel_vertical.sum()

    imagem_expandida_horizontal = np.pad(imagem.astype(float),
                                          ((0, 0), (margem, margem)),
                                          mode="edge")
    janelas_horizontais = np.lib.stride_tricks.sliding_window_view(
        imagem_expandida_horizontal, tamanho, axis=1)
    imagem_filtrada_horizontal = np.einsum("ijk,k->ij", janelas_horizontais,
                                           kernel_horizontal)

    imagem_expandida_vertical = np.pad(imagem_filtrada_horizontal,
                                        ((margem, margem), (0, 0)),
                                        mode="edge")
    janelas_verticais = np.lib.stride_tricks.sliding_window_view(
        imagem_expandida_vertical, tamanho, axis=0)

    return np.einsum("ijk,k->ij", janelas_verticais, kernel_vertical)


def criar_esboco(imagem: np.ndarray, imagem_desfocada: np.ndarray):
    """Divide a imagem original pela desfocada para realçar seus contornos."""
    menor_divisor = 1e-6
    divisor_seguro = np.maximum(imagem_desfocada, menor_divisor)

    imagem_dividida = imagem.astype(float) / divisor_seguro
    imagem_esboco = imagem_dividida * 255
    imagem_limitada = np.clip(imagem_esboco, 0, 255)

    return np.rint(imagem_limitada).astype(np.uint8)


def salvar_comparacao(original: np.ndarray, desfocada: np.ndarray,
                      esboco: np.ndarray, tamanho_kernel: int, sigma: float,
                      saida: Path):
    """Exibe e salva a sequência de etapas usada para criar o esboço."""
    saida.parent.mkdir(parents=True, exist_ok=True)

    figura, eixos = plt.subplots(1, 3, figsize=(14, 5))
    figura.subplots_adjust(wspace=0.04, bottom=0.16)
    e0, e1, e2 = eixos

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(desfocada, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(esboco, cmap="gray", vmin=0, vmax=255); e2.axis("off")

    e0.text(0.5, -0.08, "(a) imagem original", transform=e0.transAxes,
            ha="center", va="top", fontsize=11)
    e1.text(0.5, -0.08, f"(b) desfoque gaussiano {tamanho_kernel} × {tamanho_kernel} (σ = {sigma:g})",
            transform=e1.transAxes, ha="center", va="top", fontsize=11)
    e2.text(0.5, -0.08, "(c) esboço a lápis", transform=e2.transAxes,
            ha="center", va="top", fontsize=11)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    tamanho_kernel = 21
    sigma = 3.5

    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)
    kernel_gaussiano = criar_kernel_gaussiano(tamanho_kernel, sigma)
    desfocada = correlacionar_gaussiano_separavel(original, kernel_gaussiano)
    esboco = criar_esboco(original, desfocada)

    salvar_comparacao(original, desfocada, esboco, tamanho_kernel, sigma,
                      ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
