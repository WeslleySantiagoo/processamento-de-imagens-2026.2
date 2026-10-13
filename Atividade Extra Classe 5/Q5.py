from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens" / "binarias_diversas" / "beetle-15.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q5_resultados.png"


def carregar_imagem_binaria(caminho):
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")

    return imagem > 127


def criar_janelas(imagem, tamanho):
    margem = tamanho // 2
    imagem_preenchida = np.pad(imagem, margem, mode="constant", constant_values=False)
    return sliding_window_view(imagem_preenchida, (tamanho, tamanho))


def erodir(imagem, elemento_estruturante):
    if not np.any(elemento_estruturante):
        raise ValueError("O elemento estruturante deve ter ao menos uma posição marcada.")

    tamanho = elemento_estruturante.shape[0]
    janelas = criar_janelas(imagem, tamanho)
    return np.all(janelas[:, :, elemento_estruturante], axis=2)


def aplicar_hit_or_miss(imagem, elemento_objeto, elemento_fundo):
    acertos_objeto = erodir(imagem, elemento_objeto)
    acertos_fundo = erodir(~imagem, elemento_fundo)
    return acertos_objeto & acertos_fundo


def criar_padrao_cruz_3x3():
    padrao = np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ]
    )
    return padrao, ~padrao


def criar_padrao_cruz_5x5():
    padrao = np.array(
        [
            [False, False, True, False, False],
            [False, True, True, True, False],
            [True, True, True, True, True],
            [False, True, True, True, False],
            [False, False, True, False, False],
        ]
    )
    return padrao, ~padrao


def exibir_padrao(eixo, elemento_objeto, titulo):
    eixo.imshow(elemento_objeto, cmap="gray", vmin=False, vmax=True)
    eixo.set_title(titulo)
    eixo.set_xticks([])
    eixo.set_yticks([])


def exibir_resultado(eixo, resultado, titulo):
    eixo.imshow(resultado, cmap="gray", vmin=False, vmax=True)
    eixo.set_title(titulo)
    eixo.axis("off")


def criar_previa(imagem, padrao_cruz_3x3, padrao_cruz_5x5, resultado_cruz_3x3, resultado_cruz_5x5):
    # Função para criar a prévia dos resultados da transformada 
    figura, eixos = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

    exibir_resultado(eixos[0, 0], imagem, "Imagem original")
    exibir_padrao(eixos[0, 1], padrao_cruz_3x3, "Padrão 1: cruz 3 * 3")
    exibir_padrao(eixos[0, 2], padrao_cruz_5x5, "Padrão 2: cruz 5 * 5")
    exibir_resultado(eixos[1, 0], resultado_cruz_3x3, "Detecções do padrão 1")
    exibir_resultado(eixos[1, 1], resultado_cruz_5x5, "Detecções do padrão 2")

    eixos[1, 2].imshow(imagem, cmap="gray", vmin=False, vmax=True)
    linhas_cruz_3x3, colunas_cruz_3x3 = np.where(resultado_cruz_3x3)
    linhas_cruz_5x5, colunas_cruz_5x5 = np.where(resultado_cruz_5x5)
    eixos[1, 2].scatter(colunas_cruz_3x3, linhas_cruz_3x3, s=50, facecolors="none", edgecolors="red", label="Cruz 3 * 3")
    eixos[1, 2].scatter(colunas_cruz_5x5, linhas_cruz_5x5, s=70, facecolors="none", edgecolors="cyan", label="Cruz 5 * 5")
    eixos[1, 2].set_title("Posições encontradas")
    eixos[1, 2].legend(loc="lower right", fontsize=8)
    eixos[1, 2].axis("off")

    ARQUIVO_SAIDA.parent.mkdir(exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=160)
    plt.show()


def main():
    imagem = carregar_imagem_binaria(ARQUIVO_ENTRADA)
    objeto_cruz_3x3, fundo_cruz_3x3 = criar_padrao_cruz_3x3()
    objeto_cruz_5x5, fundo_cruz_5x5 = criar_padrao_cruz_5x5()

    resultado_cruz_3x3 = aplicar_hit_or_miss(imagem, objeto_cruz_3x3, fundo_cruz_3x3)
    resultado_cruz_5x5 = aplicar_hit_or_miss(imagem, objeto_cruz_5x5, fundo_cruz_5x5)

    if resultado_cruz_3x3.shape != imagem.shape or resultado_cruz_5x5.shape != imagem.shape:
        raise ValueError("A Transformada Hit-or-Miss alterou as dimensões da imagem.")

    if np.any(resultado_cruz_3x3 & ~imagem) or np.any(resultado_cruz_5x5 & ~imagem):
        raise ValueError("Uma detecção Hit-or-Miss deve ocorrer sobre pixels do objeto.")

    print(f"Imagem utilizada: {ARQUIVO_ENTRADA.name}")
    print(f"Padrão 1 - cruz 3x3: {resultado_cruz_3x3.sum()} posição(ões)")
    print(f"Padrão 2 - cruz 5x5: {resultado_cruz_5x5.sum()} posição(ões)")
    coordenadas_cruz_3x3 = [(int(linha), int(coluna)) for linha, coluna in np.argwhere(resultado_cruz_3x3)]
    coordenadas_cruz_5x5 = [(int(linha), int(coluna)) for linha, coluna in np.argwhere(resultado_cruz_5x5)]
    print(f"Coordenadas do padrão 1 (linha, coluna): {coordenadas_cruz_3x3}")
    print(f"Coordenadas do padrão 2 (linha, coluna): {coordenadas_cruz_5x5}")

    criar_previa(
        imagem,
        objeto_cruz_3x3,
        objeto_cruz_5x5,
        resultado_cruz_3x3,
        resultado_cruz_5x5,
    )


if __name__ == "__main__":
    main()
