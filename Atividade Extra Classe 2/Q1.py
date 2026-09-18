from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "city.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q1_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega a imagem como matriz uint8 de uma única banda."""
    imagem = plt.imread(caminho)

    if np.issubdtype(imagem.dtype, np.floating):
        imagem = np.rint(imagem * 255).astype(np.uint8)

    if imagem.ndim == 3:
        imagem = imagem[..., 0]

    return imagem


def negativo(imagem: np.ndarray):
    """(b) Calcula o negativo da imagem."""
    return 255 - imagem


def espelhar_verticalmente(imagem: np.ndarray):
    """(c) Espelha a imagem de cima para baixo."""
    return imagem[::-1, :]


def converter_para_100_200(imagem: np.ndarray):
    """(d) Converte linearmente as intensidades de [0, 255] para [100, 200]."""
    limite_inferior = 100
    limite_superior = 200
    intensidade_maxima_original = 255

    imagem_normalizada = imagem.astype(float) / intensidade_maxima_original

    nova_amplitude = limite_superior - limite_inferior
    imagem_transformada = limite_inferior + imagem_normalizada * nova_amplitude

    return np.rint(imagem_transformada).astype(np.uint8)


def inverter_linhas_pares(imagem: np.ndarray):
    """(e) Inverte horizontalmente apenas as linhas de índice par."""
    resultado = imagem.copy()
    resultado[::2, :] = imagem[::2, ::-1]
    return resultado


def refletir_metade_superior(imagem: np.ndarray):
    """(f) Copia a metade superior espelhada para a metade inferior."""
    resultado = imagem.copy()
    metade_inferior = imagem.shape[0] // 2
    resultado[-metade_inferior:, :] = imagem[:metade_inferior, :][::-1, :]
    return resultado


def salvar_comparacao(imagens: tuple[np.ndarray, ...], saida: Path):
    """Exibe e salva as seis imagens em uma grade 2 × 3."""
    titulos = (
        "(a) imagem original",
        "(b) negativo da imagem",
        "(c) espelhamento vertical",
        "(d) intensidades em [100, 200]",
        "(e) linhas pares invertidas",
        "(f) reflexão da metade superior",
    )

    figura, eixos = plt.subplots(2, 3, figsize=(13, 8))
    figura.subplots_adjust(hspace=0.22)
    e0, e1, e2, e3, e4, e5 = eixos.flat
    i0, i1, i2, i3, i4, i5 = imagens
    t0, t1, t2, t3, t4, t5 = titulos

    e0.imshow(i0, cmap="gray", vmin=0, vmax=255); e0.set_title(t0); e0.axis("off")
    e1.imshow(i1, cmap="gray", vmin=0, vmax=255); e1.set_title(t1); e1.axis("off")
    e2.imshow(i2, cmap="gray", vmin=0, vmax=255); e2.set_title(t2); e2.axis("off")
    e3.imshow(i3, cmap="gray", vmin=0, vmax=255); e3.set_title(t3); e3.axis("off")
    e4.imshow(i4, cmap="gray", vmin=0, vmax=255); e4.set_title(t4); e4.axis("off")
    e5.imshow(i5, cmap="gray", vmin=0, vmax=255); e5.set_title(t5); e5.axis("off")

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)
    resultados = (
        negativo(original),
        espelhar_verticalmente(original),
        converter_para_100_200(original),
        inverter_linhas_pares(original),
        refletir_metade_superior(original),
    )
    salvar_comparacao((original, *resultados), ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
