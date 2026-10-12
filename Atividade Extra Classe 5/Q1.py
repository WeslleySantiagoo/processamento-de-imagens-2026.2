from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_PRINCIPAL = PASTA_ATUAL / "imagens" / "histograma" / "city_1.png"
ARQUIVO_COMPARACAO = PASTA_ATUAL / "imagens" / "histograma" / "cameraman_1.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q1_resultados.png"


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


def normalizar_histograma(histograma, total_pixels):
    return histograma / total_pixels


def intervalo_com_maior_concentracao(histograma, largura=16):
    soma_por_faixa = np.convolve(histograma, np.ones(largura, dtype=np.int64), mode="valid")
    inicio = int(np.argmax(soma_por_faixa))
    return inicio, inicio + largura - 1, int(soma_por_faixa[inicio])


def exibir_histograma(eixo, histograma, titulo, normalizado=False):
    eixo.bar(np.arange(256), histograma, width=1.0, color="dimgray")
    eixo.set_title(titulo)
    eixo.set_xlabel("Intensidade")
    eixo.set_ylabel("Probabilidade" if normalizado else "Quantidade de pixels")
    eixo.set_xlim(0, 255)
    eixo.grid(axis="y", alpha=0.25)


def criar_previa(imagem_principal, histograma_principal, normalizado_principal,
                  imagem_comparacao, histograma_comparacao, normalizado_comparacao):
    # Função riada com auxilio de IA para melhor adequação da visualização da saída
    figura, eixos = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    eixos[0, 0].imshow(imagem_principal, cmap="gray", vmin=0, vmax=255)
    eixos[0, 0].set_title("Imagem principal: city_1.png")
    eixos[0, 0].axis("off")
    exibir_histograma(eixos[0, 1], histograma_principal, "Histograma")
    exibir_histograma(eixos[0, 2], normalizado_principal, "Histograma normalizado", True)

    eixos[1, 0].imshow(imagem_comparacao, cmap="gray", vmin=0, vmax=255)
    eixos[1, 0].set_title("Comparação: cameraman_1.png")
    eixos[1, 0].axis("off")
    exibir_histograma(eixos[1, 1], histograma_comparacao, "Histograma")
    exibir_histograma(eixos[1, 2], normalizado_comparacao, "Histograma normalizado", True)

    ARQUIVO_SAIDA.parent.mkdir(exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=160)
    plt.show()


def main():
    imagem_principal = carregar_imagem(ARQUIVO_PRINCIPAL)
    imagem_comparacao = carregar_imagem(ARQUIVO_COMPARACAO)

    histograma_principal = calcular_histograma(imagem_principal)
    histograma_comparacao = calcular_histograma(imagem_comparacao)

    total_principal = imagem_principal.size
    total_comparacao = imagem_comparacao.size
    normalizado_principal = normalizar_histograma(histograma_principal, total_principal)
    normalizado_comparacao = normalizar_histograma(histograma_comparacao, total_comparacao)

    assert histograma_principal.sum() == total_principal
    assert histograma_comparacao.sum() == total_comparacao
    assert np.isclose(normalizado_principal.sum(), 1.0)
    assert np.isclose(normalizado_comparacao.sum(), 1.0)

    faixa_principal = intervalo_com_maior_concentracao(histograma_principal)
    faixa_comparacao = intervalo_com_maior_concentracao(histograma_comparacao)

    print(f"city_1.png: dimensões={imagem_principal.shape}, pixels={total_principal}")
    print(f"city_1.png: média={imagem_principal.mean():.2f}, desvio={imagem_principal.std():.2f}")
    print(f"city_1.png: maior faixa={faixa_principal[0]}-{faixa_principal[1]} ({faixa_principal[2]} pixels)")
    print(f"cameraman_1.png: dimensões={imagem_comparacao.shape}, pixels={total_comparacao}")
    print(f"cameraman_1.png: média={imagem_comparacao.mean():.2f}, desvio={imagem_comparacao.std():.2f}")
    print(f"cameraman_1.png: maior faixa={faixa_comparacao[0]}-{faixa_comparacao[1]} ({faixa_comparacao[2]} pixels)")

    criar_previa(
        imagem_principal,
        histograma_principal,
        normalizado_principal,
        imagem_comparacao,
        histograma_comparacao,
        normalizado_comparacao,
    )


if __name__ == "__main__":
    main()
