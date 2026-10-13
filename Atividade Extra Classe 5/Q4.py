from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens" / "binarias_diversas" / "bell-1.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q4_resultados.png"
TAMANHOS_ELEMENTO = [3, 5]


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


def abrir(imagem, elemento_estruturante):
    imagem_erodida = erodir(imagem, elemento_estruturante)
    return dilatar(imagem_erodida, elemento_estruturante)


def fechar(imagem, elemento_estruturante):
    imagem_dilatada = dilatar(imagem, elemento_estruturante)
    return erodir(imagem_dilatada, elemento_estruturante)


def exibir_imagem(eixo, imagem, titulo):
    eixo.imshow(imagem, cmap="gray", vmin=False, vmax=True)
    eixo.set_title(titulo)
    eixo.axis("off")


def criar_previa(imagem, aberturas, fechamentos):
    # Previa criada com auxilio de IA para melhor visualização das imagens
    figura = plt.figure(figsize=(9, 12), constrained_layout=True)
    grade = figura.add_gridspec(3, 2)
    eixo_original = figura.add_subplot(grade[0, :])
    exibir_imagem(eixo_original, imagem, "Imagem original")

    for indice, tamanho in enumerate(TAMANHOS_ELEMENTO, start=1):
        eixo_abertura = figura.add_subplot(grade[indice, 0])
        eixo_fechamento = figura.add_subplot(grade[indice, 1])
        exibir_imagem(eixo_abertura, aberturas[tamanho], f"Abertura {tamanho} * {tamanho}")
        exibir_imagem(eixo_fechamento, fechamentos[tamanho], f"Fechamento {tamanho} * {tamanho}")

    ARQUIVO_SAIDA.parent.mkdir(exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=160)
    plt.show()


def main():
    imagem = carregar_imagem_binaria(ARQUIVO_ENTRADA)
    aberturas = {}
    fechamentos = {}

    for tamanho in TAMANHOS_ELEMENTO:
        elemento_estruturante = criar_elemento_estruturante(tamanho)
        aberturas[tamanho] = abrir(imagem, elemento_estruturante)
        fechamentos[tamanho] = fechar(imagem, elemento_estruturante)

        if aberturas[tamanho].shape != imagem.shape:
            raise ValueError("A abertura alterou as dimensões da imagem.")

        if fechamentos[tamanho].shape != imagem.shape:
            raise ValueError("O fechamento alterou as dimensões da imagem.")

    print(f"Imagem utilizada: {ARQUIVO_ENTRADA.name}")
    print(f"Pixels do objeto na imagem original: {imagem.sum()}")

    for tamanho in TAMANHOS_ELEMENTO:
        print(
            f"Elemento {tamanho}x{tamanho}: "
            f"abertura={aberturas[tamanho].sum()} pixels, "
            f"fechamento={fechamentos[tamanho].sum()} pixels"
        )

    criar_previa(imagem, aberturas, fechamentos)


if __name__ == "__main__":
    main()
