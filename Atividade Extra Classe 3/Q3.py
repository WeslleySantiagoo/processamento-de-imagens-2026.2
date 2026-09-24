"""Q3 — Filtragem de imagens por correlação de máscaras."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "aerial_view.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q3_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega uma imagem PNG como matriz bidimensional de intensidades."""
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")

    return imagem


def criar_kernel_caixa(tamanho: int):
    """Cria um kernel de caixa quadrado normalizado."""
    if tamanho <= 0 or tamanho % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser um número ímpar positivo.")

    quantidade_elementos = tamanho * tamanho
    kernel = np.ones((tamanho, tamanho), dtype=float)

    return kernel / quantidade_elementos


def correlacionar(imagem: np.ndarray, kernel: np.ndarray):
    """Aplica correlação manual vetorizada entre uma imagem e um kernel ímpar."""
    altura_kernel, largura_kernel = kernel.shape

    if altura_kernel % 2 == 0 or largura_kernel % 2 == 0:
        raise ValueError("A altura e a largura do kernel devem ser ímpares.")

    margem_vertical = altura_kernel // 2
    margem_horizontal = largura_kernel // 2

    imagem_expandida = np.pad(imagem.astype(float),
                               ((margem_vertical, margem_vertical),
                                (margem_horizontal, margem_horizontal)),
                               mode="edge")

    janelas = np.lib.stride_tricks.sliding_window_view(imagem_expandida,
                                                        kernel.shape)

    return np.einsum("ijkl,kl->ij", janelas, kernel)


def converter_para_uint8(imagem: np.ndarray):
    """Arredonda e limita a imagem filtrada ao intervalo de intensidades válido."""
    imagem_limitada = np.clip(imagem, 0, 255)
    return np.rint(imagem_limitada).astype(np.uint8)


def salvar_comparacao(original: np.ndarray, caixa_3: np.ndarray,
                      caixa_5: np.ndarray, caixa_7: np.ndarray,
                      gaussiano_3: np.ndarray, gaussiano_5: np.ndarray,
                      gaussiano_7: np.ndarray, saida: Path):
    """Exibe e salva a imagem original e os seis resultados de filtragem."""
    # Usado IA apenas para amostragem visual
    saida.parent.mkdir(parents=True, exist_ok=True)

    figura = plt.figure(figsize=(10, 11))
    grade = figura.add_gridspec(3, 6, hspace=0.2, wspace=0.02)

    e0 = figura.add_subplot(grade[0, 2:4])
    e1 = figura.add_subplot(grade[1, 0:2])
    e2 = figura.add_subplot(grade[1, 2:4])
    e3 = figura.add_subplot(grade[1, 4:6])
    e4 = figura.add_subplot(grade[2, 0:2])
    e5 = figura.add_subplot(grade[2, 2:4])
    e6 = figura.add_subplot(grade[2, 4:6])

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(caixa_3, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(caixa_5, cmap="gray", vmin=0, vmax=255); e2.axis("off")
    e3.imshow(caixa_7, cmap="gray", vmin=0, vmax=255); e3.axis("off")
    e4.imshow(gaussiano_3, cmap="gray", vmin=0, vmax=255); e4.axis("off")
    e5.imshow(gaussiano_5, cmap="gray", vmin=0, vmax=255); e5.axis("off")
    e6.imshow(gaussiano_7, cmap="gray", vmin=0, vmax=255); e6.axis("off")

    e0.text(0.5, -0.08, "(a) imagem original", transform=e0.transAxes,
            ha="center", va="top", fontsize=11)
    e1.text(0.5, -0.08, "(b) caixa 3 * 3", transform=e1.transAxes,
            ha="center", va="top", fontsize=11)
    e2.text(0.5, -0.08, "(c) caixa 5 * 5", transform=e2.transAxes,
            ha="center", va="top", fontsize=11)
    e3.text(0.5, -0.08, "(d) caixa 7 * 7", transform=e3.transAxes,
            ha="center", va="top", fontsize=11)
    e4.text(0.5, -0.08, "(e) gaussiano 3 * 3", transform=e4.transAxes,
            ha="center", va="top", fontsize=11)
    e5.text(0.5, -0.08, "(f) gaussiano 5 * 5", transform=e5.transAxes,
            ha="center", va="top", fontsize=11)
    e6.text(0.5, -0.08, "(g) gaussiano 7 * 7", transform=e6.transAxes,
            ha="center", va="top", fontsize=11)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    kernel_caixa_3 = criar_kernel_caixa(3)
    kernel_caixa_5 = criar_kernel_caixa(5)
    kernel_caixa_7 = criar_kernel_caixa(7)

    kernel_gaussiano_3 = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ], dtype=float) / 16

    kernel_gaussiano_5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ], dtype=float) / 273

    kernel_gaussiano_7 = np.array([
        [0, 0, 1, 2, 1, 0, 0],
        [0, 3, 13, 22, 13, 3, 0],
        [1, 13, 59, 97, 59, 13, 1],
        [2, 22, 97, 159, 97, 22, 2],
        [1, 13, 59, 97, 59, 13, 1],
        [0, 3, 13, 22, 13, 3, 0],
        [0, 0, 1, 2, 1, 0, 0],
    ], dtype=float) / 1003

    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)

    caixa_3 = converter_para_uint8(correlacionar(original, kernel_caixa_3))
    caixa_5 = converter_para_uint8(correlacionar(original, kernel_caixa_5))
    caixa_7 = converter_para_uint8(correlacionar(original, kernel_caixa_7))
    gaussiano_3 = converter_para_uint8(correlacionar(original, kernel_gaussiano_3))
    gaussiano_5 = converter_para_uint8(correlacionar(original, kernel_gaussiano_5))
    gaussiano_7 = converter_para_uint8(correlacionar(original, kernel_gaussiano_7))

    salvar_comparacao(original, caixa_3, caixa_5, caixa_7,
                      gaussiano_3, gaussiano_5, gaussiano_7, ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
