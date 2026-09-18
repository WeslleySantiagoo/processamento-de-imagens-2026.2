"""Q4 — Extração dos planos de bits de uma imagem monocromática."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "baboon_monocromatica.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q4_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega a imagem como matriz de intensidades inteiras entre 0 e 255."""
    imagem = plt.imread(caminho)

    if np.issubdtype(imagem.dtype, np.floating):
        imagem = np.rint(imagem * 255).astype(np.uint8)

    if imagem.ndim == 3:
        imagem = imagem[..., 0]

    return imagem


def extrair_plano_de_bit(imagem: np.ndarray, ordem_bit: int):
    """Extrai um plano de bit de ordem 0 a 7 e o representa com 0 e 255."""
    if ordem_bit < 0 or ordem_bit > 7:
        raise ValueError("A ordem do plano de bit deve estar no intervalo de 0 a 7.")

    # Desloca o bit desejado para a posição menos significativa e isola-o com 1.
    bits_deslocados = imagem >> ordem_bit
    plano_com_valores_0_1 = bits_deslocados & 1

    # Converte os valores binários 0 e 1 em intensidades visíveis 0 e 255.
    plano_de_bit = plano_com_valores_0_1 * 255

    return plano_de_bit.astype(np.uint8)


def salvar_comparacao(original: np.ndarray, plano_1: np.ndarray,
                      plano_2: np.ndarray, plano_3: np.ndarray,
                      ordem_bit_1: int, ordem_bit_2: int, ordem_bit_3: int,
                      saida: Path):
    """Exibe e salva a imagem original ao lado de três planos de bits."""
    # Usado IA apenas para amostragem visual

    figura, eixos = plt.subplots(1, 4, figsize=(14, 4))
    figura.subplots_adjust(wspace=0.08, bottom=0.14)

    e0, e1, e2, e3 = eixos

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(plano_1, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(plano_2, cmap="gray", vmin=0, vmax=255); e2.axis("off")
    e3.imshow(plano_3, cmap="gray", vmin=0, vmax=255); e3.axis("off")

    e0.text(0.5, -0.08, "imagem", transform=e0.transAxes,
            ha="center", va="top", fontsize=12)
    e1.text(0.5, -0.08, f"plano de bit {ordem_bit_1}", transform=e1.transAxes,
            ha="center", va="top", fontsize=12)
    e2.text(0.5, -0.08, f"plano de bit {ordem_bit_2}", transform=e2.transAxes,
            ha="center", va="top", fontsize=12)
    e3.text(0.5, -0.08, f"plano de bit {ordem_bit_3}", transform=e3.transAxes,
            ha="center", va="top", fontsize=12)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    ordem_bit_1 = 0
    ordem_bit_2 = 4
    ordem_bit_3 = 7

    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)

    plano_1 = extrair_plano_de_bit(original, ordem_bit_1)
    plano_2 = extrair_plano_de_bit(original, ordem_bit_2)
    plano_3 = extrair_plano_de_bit(original, ordem_bit_3)

    salvar_comparacao(original, plano_1, plano_2, plano_3,
                      ordem_bit_1, ordem_bit_2, ordem_bit_3, ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
