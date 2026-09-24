"""Q5 — Detecção de bordas com filtro Laplaciano."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "gray_peppers.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q5_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega uma imagem PNG como matriz bidimensional de intensidades."""
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")

    return imagem


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


def visualizar_resposta_com_sinal(resposta: np.ndarray):
    """Mostra valores negativos escuros, positivos claros e zero em cinza médio."""
    maior_modulo = np.max(np.abs(resposta))

    if maior_modulo == 0:
        return np.full(resposta.shape, 127, dtype=np.uint8)

    resposta_escalada = 127.5 + resposta * (127.5 / maior_modulo)
    return np.rint(np.clip(resposta_escalada, 0, 255)).astype(np.uint8)


def visualizar_bordas(resposta: np.ndarray):
    """Converte o módulo da resposta laplaciana em uma imagem de bordas."""
    maior_modulo = np.max(np.abs(resposta))

    if maior_modulo == 0:
        return np.zeros(resposta.shape, dtype=np.uint8)

    bordas = np.abs(resposta) * (255 / maior_modulo)
    return np.rint(np.clip(bordas, 0, 255)).astype(np.uint8)


def salvar_comparacao(original: np.ndarray, resposta_assinada: np.ndarray,
                      bordas: np.ndarray, saida: Path):
    """Exibe e salva a imagem original, a resposta e as bordas laplacianas."""
    saida.parent.mkdir(parents=True, exist_ok=True)

    figura, eixos = plt.subplots(1, 3, figsize=(12, 5))
    figura.subplots_adjust(wspace=0.04, bottom=0.16)
    e0, e1, e2 = eixos

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(resposta_assinada, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(bordas, cmap="gray", vmin=0, vmax=255); e2.axis("off")

    e0.text(0.5, -0.08, "(a) imagem original", transform=e0.transAxes,
            ha="center", va="top", fontsize=11)
    e1.text(0.5, -0.08, "(b) resposta laplaciana", transform=e1.transAxes,
            ha="center", va="top", fontsize=11)
    e2.text(0.5, -0.08, "(c) bordas em módulo", transform=e2.transAxes,
            ha="center", va="top", fontsize=11)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    kernel_laplaciano = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ], dtype=float)

    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)
    resposta_laplaciana = correlacionar(original, kernel_laplaciano)
    resposta_assinada = visualizar_resposta_com_sinal(resposta_laplaciana)
    bordas = visualizar_bordas(resposta_laplaciana)

    salvar_comparacao(original, resposta_assinada, bordas, ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
