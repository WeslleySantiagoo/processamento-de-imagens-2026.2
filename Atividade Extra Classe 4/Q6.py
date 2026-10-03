"""Q6 - Realce de bordas com filtro passa-alta ideal."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "boats.pgm"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q6_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega uma imagem como matriz de intensidades em tons de cinza."""
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")

    return imagem


def criar_matriz_distancias(forma: tuple):
    """Calcula a distância de cada frequência ao centro do espectro."""
    altura, largura = forma
    linhas, colunas = np.ogrid[:altura, :largura]
    centro_linha = altura // 2
    centro_coluna = largura // 2

    distancia_quadrada = (linhas - centro_linha) ** 2
    distancia_quadrada = distancia_quadrada + (colunas - centro_coluna) ** 2

    return np.sqrt(distancia_quadrada)


def criar_mascara_passa_baixa(forma: tuple, raio: int):
    """Cria a máscara passa-baixa ideal correspondente a um raio informado."""
    if raio <= 0:
        raise ValueError("O raio deve ser maior que zero.")

    distancia = criar_matriz_distancias(forma)
    return (distancia <= raio).astype(float)


def criar_mascara_passa_alta(forma: tuple, raio: int):
    """Cria uma máscara ideal que remove as baixas frequências centrais."""
    mascara_passa_baixa = criar_mascara_passa_baixa(forma, raio)
    return 1 - mascara_passa_baixa


def aplicar_passa_alta_ideal(imagem: np.ndarray, raio: int):
    """Aplica a máscara passa-alta ideal e reconstrói a resposta filtrada."""
    mascara = criar_mascara_passa_alta(imagem.shape, raio)
    espectro = np.fft.fft2(imagem.astype(np.float32))
    espectro_centralizado = np.fft.fftshift(espectro)
    espectro_filtrado = espectro_centralizado * mascara
    espectro_original = np.fft.ifftshift(espectro_filtrado)
    imagem_filtrada = np.real(np.fft.ifft2(espectro_original))

    return mascara, imagem_filtrada


def normalizar_para_visualizacao(imagem: np.ndarray):
    """Converte uma resposta com valores negativos e positivos para 0 a 255."""
    minimo = imagem.min()
    maximo = imagem.max()

    if maximo == minimo:
        return np.zeros(imagem.shape, dtype=np.uint8)

    imagem_normalizada = (imagem - minimo) * (255 / (maximo - minimo))
    return np.rint(imagem_normalizada).astype(np.uint8)


def salvar_comparacao(original: np.ndarray, mascara_10: np.ndarray,
                      mascara_60: np.ndarray, filtrada_10: np.ndarray,
                      filtrada_60: np.ndarray, saida: Path):
    """Exibe e salva as máscaras e os resultados passa-alta para dois raios."""
    #Usado IA apenas para amostragem
    saida.parent.mkdir(parents=True, exist_ok=True)

    figura, eixos = plt.subplots(2, 3, figsize=(12, 8))
    figura.subplots_adjust(wspace=0.04, hspace=0.20, bottom=0.08)
    e0, e1, e2, e3, e4, e5 = eixos.flat

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(mascara_10, cmap="gray", vmin=0, vmax=1); e1.axis("off")
    e2.imshow(mascara_60, cmap="gray", vmin=0, vmax=1); e2.axis("off")
    e3.imshow(normalizar_para_visualizacao(filtrada_10), cmap="gray", vmin=0, vmax=255); e3.axis("off")
    e4.imshow(normalizar_para_visualizacao(filtrada_60), cmap="gray", vmin=0, vmax=255); e4.axis("off")
    e5.axis("off")

    e0.text(0.5, -0.08, "(a) imagem original", transform=e0.transAxes,
            ha="center", va="top", fontsize=11)
    e1.text(0.5, -0.08, "(b) máscara r = 10", transform=e1.transAxes,
            ha="center", va="top", fontsize=11)
    e2.text(0.5, -0.08, "(c) máscara r = 60", transform=e2.transAxes,
            ha="center", va="top", fontsize=11)
    e3.text(0.5, -0.08, "(d) passa-alta r = 10", transform=e3.transAxes,
            ha="center", va="top", fontsize=11)
    e4.text(0.5, -0.08, "(e) passa-alta r = 60", transform=e4.transAxes,
            ha="center", va="top", fontsize=11)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    raio_pequeno = 10
    raio_grande = 60

    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)
    mascara_10, filtrada_10 = aplicar_passa_alta_ideal(original, raio_pequeno)
    mascara_60, filtrada_60 = aplicar_passa_alta_ideal(original, raio_grande)

    salvar_comparacao(original, mascara_10, mascara_60, filtrada_10,
                      filtrada_60, ARQUIVO_SAIDA)
    print(f"Imagem usada: {ARQUIVO_ENTRADA.name}")
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
