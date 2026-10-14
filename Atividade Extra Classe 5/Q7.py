from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


PASTA_ATUAL = Path(__file__).resolve().parent

# Descomente, e comente para alterar entre as imagens de teste
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens" / "binarias_diversas" / "cattle-3.png"
LINHA_SEMENTE, COLUNA_SEMENTE = 432, 630
# ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens" / "binarias_diversas" / "apple-1.png"
# LINHA_SEMENTE, COLUNA_SEMENTE = 79, 91

ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q7_resultados.png"


def carregar_imagem_binaria(caminho):
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")

    return imagem > 127


def criar_elemento_estruturante():
    return np.ones((3, 3), dtype=bool)


def dilatar(imagem, elemento_estruturante):
    tamanho = elemento_estruturante.shape[0]
    margem = tamanho // 2
    imagem_preenchida = np.pad(imagem, margem, mode="constant", constant_values=False)
    janelas = sliding_window_view(imagem_preenchida, (tamanho, tamanho))
    return np.any(janelas[:, :, elemento_estruturante], axis=2)


def criar_semente(formato, linha, coluna):
    if linha < 0 or linha >= formato[0] or coluna < 0 or coluna >= formato[1]:
        raise ValueError("A semente deve estar dentro dos limites da imagem.")

    semente = np.zeros(formato, dtype=bool)
    semente[linha, coluna] = True
    return semente


def preencher_buraco(imagem, semente, elemento_estruturante):
    complemento = ~imagem
    regiao_atual = semente.copy()
    iteracoes = 0

    while True:
        regiao_expandida = dilatar(regiao_atual, elemento_estruturante) & complemento
        iteracoes += 1

        if np.array_equal(regiao_expandida, regiao_atual):
            break

        regiao_atual = regiao_expandida

    imagem_preenchida = imagem | regiao_atual
    return regiao_atual, imagem_preenchida, iteracoes


def exibir_imagem(eixo, imagem, titulo):
    eixo.imshow(imagem, cmap="gray", vmin=False, vmax=True)
    eixo.set_title(titulo)
    eixo.axis("off")


def criar_previa(imagem, semente, regiao, imagem_preenchida, iteracoes):
    # Função de amostragem criada com auxilio de IA para melhor visualização dos resultados
    figura, eixos = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)

    exibir_imagem(eixos[0, 0], imagem, "Imagem original")

    eixos[0, 1].imshow(imagem, cmap="gray", vmin=False, vmax=True)
    linhas, colunas = np.where(semente)
    eixos[0, 1].scatter(colunas, linhas, color="red", s=45, label="Semente")
    eixos[0, 1].set_title("Ponto inicial dentro do buraco")
    eixos[0, 1].legend(loc="lower right")
    eixos[0, 1].axis("off")

    exibir_imagem(eixos[1, 0], regiao, f"Região obtida em {iteracoes} iterações")
    exibir_imagem(eixos[1, 1], imagem_preenchida, "Imagem após o preenchimento")

    ARQUIVO_SAIDA.parent.mkdir(exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=160)
    plt.show()


def main():
    imagem = carregar_imagem_binaria(ARQUIVO_ENTRADA)
    elemento_estruturante = criar_elemento_estruturante()
    semente = criar_semente(imagem.shape, LINHA_SEMENTE, COLUNA_SEMENTE)

    if imagem[LINHA_SEMENTE, COLUNA_SEMENTE]:
        raise ValueError("A semente deve ser posicionada em um pixel de fundo dentro do buraco.")

    regiao, imagem_preenchida, iteracoes = preencher_buraco(imagem, semente, elemento_estruturante)

    if np.any(regiao & imagem):
        raise ValueError("A região preenchida não deve ocupar pixels do objeto original.")

    if imagem_preenchida.sum() <= imagem.sum():
        raise ValueError("O preenchimento deveria adicionar pixels ao objeto original.")

    print(f"Imagem utilizada: {ARQUIVO_ENTRADA.name}")
    print(f"Semente: linha={LINHA_SEMENTE}, coluna={COLUNA_SEMENTE}")
    print(f"Iterações até a convergência: {iteracoes}")
    print(f"Pixels preenchidos: {regiao.sum()}")

    criar_previa(imagem, semente, regiao, imagem_preenchida, iteracoes)


if __name__ == "__main__":
    main()
