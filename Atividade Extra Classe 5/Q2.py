from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens" / "histograma" / "baboon_monocromatica_1.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q2_resultados.png"


def carregar_imagem(caminho):
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")

    return imagem


def calcular_histograma(imagem):
    histograma = np.zeros(256, dtype=np.int64)

    for intensidade in imagem.ravel():
        histograma[int(intensidade)] += 1

    return histograma


def calcular_tabela_equalizacao(histograma, total_pixels):
    probabilidades = histograma / total_pixels
    cdf = np.cumsum(probabilidades)
    tabela = np.rint(255 * cdf).astype(np.uint8)
    return probabilidades, cdf, tabela


def equalizar_imagem(imagem, tabela):
    return tabela[imagem]


def exibir_histograma(eixo, histograma, titulo):
    eixo.bar(np.arange(256), histograma, width=1.0, color="dimgray")
    eixo.set_title(titulo)
    eixo.set_xlabel("Intensidade")
    eixo.set_ylabel("Quantidade de pixels")
    eixo.set_xlim(0, 255)
    eixo.grid(axis="y", alpha=0.25)


def criar_previa(imagem, histograma_original, tabela, imagem_equalizada, histograma_equalizado):
    # Função criada com auxilio de IA para melhor adequação da visualização da saída
    figura, eixos = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)

    eixos[0, 0].imshow(imagem, cmap="gray", vmin=0, vmax=255)
    eixos[0, 0].set_title("Imagem original")
    eixos[0, 0].axis("off")

    exibir_histograma(eixos[0, 1], histograma_original, "Histograma original")

    eixos[0, 2].plot(np.arange(256), tabela, color="black")
    eixos[0, 2].set_title("Função de transformação $s_k$")
    eixos[0, 2].set_xlabel("Intensidade de entrada $r_k$")
    eixos[0, 2].set_ylabel("Intensidade de saída $s_k$")
    eixos[0, 2].set_xlim(0, 255)
    eixos[0, 2].set_ylim(0, 255)
    eixos[0, 2].grid(alpha=0.25)

    eixos[1, 0].imshow(imagem_equalizada, cmap="gray", vmin=0, vmax=255)
    eixos[1, 0].set_title("Imagem equalizada")
    eixos[1, 0].axis("off")

    exibir_histograma(eixos[1, 1], histograma_equalizado, "Histograma equalizado")

    eixos[1, 2].axis("off")

    ARQUIVO_SAIDA.parent.mkdir(exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=160)
    plt.show()


def main():
    imagem = carregar_imagem(ARQUIVO_ENTRADA)
    histograma_original = calcular_histograma(imagem)
    probabilidades, cdf, tabela = calcular_tabela_equalizacao(histograma_original, imagem.size)
    imagem_equalizada = equalizar_imagem(imagem, tabela)
    histograma_equalizado = calcular_histograma(imagem_equalizada)

    assert histograma_original.sum() == imagem.size
    assert np.isclose(probabilidades.sum(), 1.0)
    assert np.all(np.diff(cdf) >= 0)
    assert tabela.min() >= 0 and tabela.max() <= 255
    assert imagem_equalizada.shape == imagem.shape

    print(f"Imagem original: média={imagem.mean():.2f}, desvio={imagem.std():.2f}")
    print(f"Imagem equalizada: média={imagem_equalizada.mean():.2f}, desvio={imagem_equalizada.std():.2f}")
    print(f"Intensidades originais: {imagem.min()} a {imagem.max()}")
    print(f"Intensidades equalizadas: {imagem_equalizada.min()} a {imagem_equalizada.max()}")
    print(f"Níveis ocupados antes: {np.count_nonzero(histograma_original)}")
    print(f"Níveis ocupados depois: {np.count_nonzero(histograma_equalizado)}")

    criar_previa(imagem, histograma_original, tabela, imagem_equalizada, histograma_equalizado)


if __name__ == "__main__":
    main()
