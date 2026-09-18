"""Q6 — Combinação de imagens por média ponderada."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_IMAGEM_A = PASTA_ATUAL / "imagens de entrada" / "baboon_monocromatica.png"
ARQUIVO_IMAGEM_B = PASTA_ATUAL / "imagens de entrada" / "butterfly.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q6_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega a imagem como matriz de intensidades inteiras entre 0 e 255."""
    imagem = plt.imread(caminho)

    if np.issubdtype(imagem.dtype, np.floating):
        imagem = np.rint(imagem * 255).astype(np.uint8)

    if imagem.ndim == 3:
        imagem = imagem[..., 0]

    return imagem


def combinar_imagens(imagem_a: np.ndarray, imagem_b: np.ndarray,
                     peso_a: float, peso_b: float):
    """Combina duas imagens por peso_a * A + peso_b * B."""
    if imagem_a.shape != imagem_b.shape:
        raise ValueError("As duas imagens devem ter o mesmo tamanho.")

    if not np.isclose(peso_a + peso_b, 1):
        raise ValueError("A soma dos pesos deve ser igual a 1.")

    imagem_combinada = imagem_a.astype(float) * peso_a
    imagem_combinada = imagem_combinada + imagem_b.astype(float) * peso_b

    return np.rint(imagem_combinada).astype(np.uint8)


def formatar_expressao(peso_a: float, peso_b: float):
    """Produz o texto da legenda conforme os pesos definidos no main."""
    # Usado IA apenas para amostragem visual
    texto_peso_a = f"{peso_a:g}".replace(".", ",")
    texto_peso_b = f"{peso_b:g}".replace(".", ",")
    return f"{texto_peso_a}*A + {texto_peso_b}*B"


def salvar_comparacao(imagem_a: np.ndarray, imagem_b: np.ndarray,
                      combinacao_1: np.ndarray, combinacao_2: np.ndarray,
                      combinacao_3: np.ndarray, peso_a_1: float, peso_b_1: float,
                      peso_a_2: float, peso_b_2: float, peso_a_3: float,
                      peso_b_3: float, saida: Path):
    """Exibe e salva as imagens de entrada e as três combinações ponderadas."""
    # Usado IA apenas para amostragem visual
    figura = plt.figure(figsize=(12, 8))
    grade = figura.add_gridspec(2, 6, hspace=0.26, wspace=0.18)

    e0 = figura.add_subplot(grade[0, 1:3])
    e1 = figura.add_subplot(grade[0, 3:5])
    e2 = figura.add_subplot(grade[1, 0:2])
    e3 = figura.add_subplot(grade[1, 2:4])
    e4 = figura.add_subplot(grade[1, 4:6])

    e0.imshow(imagem_a, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(imagem_b, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(combinacao_1, cmap="gray", vmin=0, vmax=255); e2.axis("off")
    e3.imshow(combinacao_2, cmap="gray", vmin=0, vmax=255); e3.axis("off")
    e4.imshow(combinacao_3, cmap="gray", vmin=0, vmax=255); e4.axis("off")

    e0.text(0.5, -0.08, "(a) imagem A", transform=e0.transAxes,
            ha="center", va="top", fontsize=12)
    e1.text(0.5, -0.08, "(b) imagem B", transform=e1.transAxes,
            ha="center", va="top", fontsize=12)
    e2.text(0.5, -0.08, f"(c) {formatar_expressao(peso_a_1, peso_b_1)}",
            transform=e2.transAxes, ha="center", va="top", fontsize=12)
    e3.text(0.5, -0.08, f"(d) {formatar_expressao(peso_a_2, peso_b_2)}",
            transform=e3.transAxes, ha="center", va="top", fontsize=12)
    e4.text(0.5, -0.08, f"(e) {formatar_expressao(peso_a_3, peso_b_3)}",
            transform=e4.transAxes, ha="center", va="top", fontsize=12)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    peso_a_1 = 0.2
    peso_b_1 = 0.8
    peso_a_2 = 0.5
    peso_b_2 = 0.5
    peso_a_3 = 0.8
    peso_b_3 = 0.2

    imagem_a = carregar_imagem_monocromatica(ARQUIVO_IMAGEM_A)
    imagem_b = carregar_imagem_monocromatica(ARQUIVO_IMAGEM_B)

    combinacao_1 = combinar_imagens(imagem_a, imagem_b, peso_a_1, peso_b_1)
    combinacao_2 = combinar_imagens(imagem_a, imagem_b, peso_a_2, peso_b_2)
    combinacao_3 = combinar_imagens(imagem_a, imagem_b, peso_a_3, peso_b_3)

    salvar_comparacao(imagem_a, imagem_b, combinacao_1, combinacao_2,
                      combinacao_3, peso_a_1, peso_b_1, peso_a_2, peso_b_2,
                      peso_a_3, peso_b_3, ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
