from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens" / "binarias_diversas" / "camel-6.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q6_resultados.png"
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


def erodir(imagem, elemento_estruturante):
    tamanho = elemento_estruturante.shape[0]
    margem = tamanho // 2
    imagem_preenchida = np.pad(imagem, margem, mode="constant", constant_values=False)
    janelas = sliding_window_view(imagem_preenchida, (tamanho, tamanho))
    return np.all(janelas[:, :, elemento_estruturante], axis=2)


def extrair_borda_interna(imagem, elemento_estruturante):
    imagem_erodida = erodir(imagem, elemento_estruturante)
    borda = imagem & ~imagem_erodida
    return imagem_erodida, borda


def exibir_imagem(eixo, imagem, titulo):
    eixo.imshow(imagem, cmap="gray", vmin=False, vmax=True)
    eixo.set_title(titulo)
    eixo.axis("off")


def criar_previa(imagem, imagens_erodidas, bordas):
    figura = plt.figure(figsize=(11, 12), constrained_layout=True)
    grade = figura.add_gridspec(3, 3)

    eixo_original = figura.add_subplot(grade[0, :])
    exibir_imagem(eixo_original, imagem, "Imagem original")

    for indice, tamanho in enumerate(TAMANHOS_ELEMENTO):
        eixo_erosao = figura.add_subplot(grade[1, indice])
        eixo_borda = figura.add_subplot(grade[2, indice])
        exibir_imagem(eixo_erosao, imagens_erodidas[tamanho], f"Erosão {tamanho} * {tamanho}")
        exibir_imagem(eixo_borda, bordas[tamanho], f"Borda interna {tamanho} * {tamanho}")

    ARQUIVO_SAIDA.parent.mkdir(exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=160)
    plt.show()


def main():
    imagem = carregar_imagem_binaria(ARQUIVO_ENTRADA)
    imagens_erodidas = {}
    bordas = {}

    for tamanho in TAMANHOS_ELEMENTO:
        elemento_estruturante = criar_elemento_estruturante(tamanho)
        imagens_erodidas[tamanho], bordas[tamanho] = extrair_borda_interna(imagem, elemento_estruturante)

        if imagens_erodidas[tamanho].sum() > imagem.sum():
            raise ValueError("A erosão não pode aumentar a quantidade de pixels do objeto.")

        if np.any(bordas[tamanho] & ~imagem):
            raise ValueError("A borda interna deve estar contida na imagem original.")

    print(f"Imagem utilizada: {ARQUIVO_ENTRADA.name}")
    print(f"Pixels do objeto na imagem original: {imagem.sum()}")

    for tamanho in TAMANHOS_ELEMENTO:
        print(
            f"Elemento {tamanho}x{tamanho}: "
            f"objeto erodido={imagens_erodidas[tamanho].sum()} pixels, "
            f"borda={bordas[tamanho].sum()} pixels"
        )

    criar_previa(imagem, imagens_erodidas, bordas)


if __name__ == "__main__":
    main()
