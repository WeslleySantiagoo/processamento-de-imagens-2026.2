"""Q3 — Binarização por limiar fixo."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "baboon_monocromatica.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q3_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega a imagem como matriz de intensidades inteiras entre 0 e 255."""
    imagem = plt.imread(caminho)

    if np.issubdtype(imagem.dtype, np.floating):
        imagem = np.rint(imagem * 255).astype(np.uint8)

    if imagem.ndim == 3:
        imagem = imagem[..., 0]

    return imagem


def binarizar_por_limiar(imagem: np.ndarray, limiar: int):
    """Retorna 255 para pixels maiores que o limiar e 0 para os demais."""
    if limiar < 0 or limiar > 255:
        raise ValueError("O limiar deve estar no intervalo de 0 a 255.")

    mascara_acima_do_limiar = imagem > limiar
    imagem_binaria = mascara_acima_do_limiar.astype(np.uint8) * 255

    return imagem_binaria


def salvar_comparacao(original: np.ndarray, binaria: np.ndarray,
                      limiar: int, saida: Path):
    """Exibe e salva a imagem original e sua versão binarizada."""
    # Usado IA apenas para amostragem visual

    figura, eixos = plt.subplots(1, 2, figsize=(8, 4.8))
    figura.subplots_adjust(wspace=0.08, bottom=0.14)

    e0, e1 = eixos

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(binaria, cmap="gray", vmin=0, vmax=255); e1.axis("off")

    e0.text(0.5, -0.08, "imagem monocromática", transform=e0.transAxes,
            ha="center", va="top", fontsize=12)
    e1.text(0.5, -0.08, f"imagem binária ($T$ = {limiar})", transform=e1.transAxes,
            ha="center", va="top", fontsize=12)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    limiar = 128

    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)
    imagem_binaria = binarizar_por_limiar(original, limiar)

    salvar_comparacao(original, imagem_binaria, limiar, ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
