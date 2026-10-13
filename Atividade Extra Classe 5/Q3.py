from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens" / "binarias_diversas" / "brick-01.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q3_resultados.png"
TAMANHOS_ELEMENTO = [3, 5, 9]


def carregar_imagem_binaria(caminho):
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")

    return imagem > 127


def criar_elemento_estruturante(tamanho):
    if tamanho <= 0 or tamanho % 2 == 0:
        raise ValueError("O tamanho do elemento estruturante deve ser positivo e ímpar.")

    return np.ones((tamanho, tamanho), dtype=bool)


def criar_janelas(imagem, tamanho):
    margem = tamanho // 2
    imagem_preenchida = np.pad(imagem, margem, mode="constant", constant_values=False)
    return sliding_window_view(imagem_preenchida, (tamanho, tamanho))


def erodir(imagem, elemento_estruturante):
    tamanho = elemento_estruturante.shape[0]
    janelas = criar_janelas(imagem, tamanho)
    return np.all(janelas[:, :, elemento_estruturante], axis=2)


def dilatar(imagem, elemento_estruturante):
    tamanho = elemento_estruturante.shape[0]
    janelas = criar_janelas(imagem, tamanho)
    return np.any(janelas[:, :, elemento_estruturante], axis=2)


def exibir_imagem(eixo, imagem, titulo):
    eixo.imshow(imagem, cmap="gray", vmin=False, vmax=True)
    eixo.set_title(titulo)
    eixo.axis("off")


def criar_previa(imagem, erosoes, dilatacoes):
    # Função criada com auxílio de IA para melhor adequação da visualização da saída
    figura, eixos = plt.subplots(3, 3, figsize=(11, 12), constrained_layout=True)

    eixos[0, 0].axis("off")
    exibir_imagem(eixos[0, 1], imagem, "Imagem original")
    eixos[0, 2].axis("off")

    for indice, tamanho in enumerate(TAMANHOS_ELEMENTO):
        exibir_imagem(eixos[1, indice], erosoes[tamanho], f"Erosão - {tamanho} * {tamanho}")
        exibir_imagem(eixos[2, indice], dilatacoes[tamanho], f"Dilatação - {tamanho} * {tamanho}")

    ARQUIVO_SAIDA.parent.mkdir(exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=160)
    plt.show()


def main():
    imagem = carregar_imagem_binaria(ARQUIVO_ENTRADA)
    erosoes = {}
    dilatacoes = {}

    for tamanho in TAMANHOS_ELEMENTO:
        elemento_estruturante = criar_elemento_estruturante(tamanho)
        erosoes[tamanho] = erodir(imagem, elemento_estruturante)
        dilatacoes[tamanho] = dilatar(imagem, elemento_estruturante)

        if erosoes[tamanho].sum() > imagem.sum():
            raise ValueError("A erosão não pode aumentar a quantidade de pixels do objeto.")

        if dilatacoes[tamanho].sum() < imagem.sum():
            raise ValueError("A dilatação não pode reduzir a quantidade de pixels do objeto.")

    print(f"Imagem utilizada: {ARQUIVO_ENTRADA.name}")
    print(f"Pixels do objeto na imagem original: {imagem.sum()}")

    for tamanho in TAMANHOS_ELEMENTO:
        print(
            f"Elemento {tamanho}x{tamanho}: "
            f"erosão={erosoes[tamanho].sum()} pixels, "
            f"dilatação={dilatacoes[tamanho].sum()} pixels"
        )

    criar_previa(imagem, erosoes, dilatacoes)


if __name__ == "__main__":
    main()
