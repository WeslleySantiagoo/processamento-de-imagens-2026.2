"""Q5 - Filtro Gaussiano no domínio da frequência."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "cameraman.tif"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q5_resultados.png"


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


def criar_mascara_gaussiana(forma: tuple, sigma: float):
    """Cria do zero uma máscara passa-baixa gaussiana no domínio da frequência."""
    if sigma <= 0:
        raise ValueError("O valor de sigma deve ser maior que zero.")

    distancia = criar_matriz_distancias(forma)
    expoente = -(distancia ** 2) / (2 * sigma ** 2)

    return np.exp(expoente)


def criar_mascara_passa_baixa_ideal(forma: tuple, raio: int):
    """Cria uma máscara circular passa-baixa ideal para comparação."""
    if raio <= 0:
        raise ValueError("O raio deve ser maior que zero.")

    distancia = criar_matriz_distancias(forma)
    return (distancia <= raio).astype(float)


def aplicar_mascara_no_espectro(imagem: np.ndarray, mascara: np.ndarray):
    """Aplica uma máscara centralizada e reconstrói a imagem filtrada."""
    if imagem.shape != mascara.shape:
        raise ValueError("A máscara deve ter o mesmo tamanho da imagem.")

    espectro = np.fft.fft2(imagem.astype(np.float32))
    espectro_centralizado = np.fft.fftshift(espectro)
    espectro_filtrado = espectro_centralizado * mascara
    espectro_original = np.fft.ifftshift(espectro_filtrado)

    return np.real(np.fft.ifft2(espectro_original))


def converter_para_uint8(imagem: np.ndarray):
    """Arredonda e limita uma imagem reconstruída ao intervalo de 0 a 255."""
    return np.rint(np.clip(imagem, 0, 255)).astype(np.uint8)


def salvar_comparacao(original: np.ndarray, gaussiana_10: np.ndarray,
                      ideal_10: np.ndarray, gaussiana_60: np.ndarray,
                      ideal_60: np.ndarray, saida: Path):
    """Exibe e salva a comparação entre filtros gaussianos e ideais."""
    # Usado IA apenas para amostragem
    saida.parent.mkdir(parents=True, exist_ok=True)

    figura, eixos = plt.subplots(2, 3, figsize=(12, 8))
    figura.subplots_adjust(wspace=0.04, hspace=0.20, bottom=0.08)
    e0, e1, e2, e3, e4, e5 = eixos.flat

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(gaussiana_10, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(ideal_10, cmap="gray", vmin=0, vmax=255); e2.axis("off")
    e3.imshow(gaussiana_60, cmap="gray", vmin=0, vmax=255); e3.axis("off")
    e4.imshow(ideal_60, cmap="gray", vmin=0, vmax=255); e4.axis("off")
    e5.axis("off")

    e0.text(0.5, -0.08, "(a) imagem original", transform=e0.transAxes,
            ha="center", va="top", fontsize=11)
    e1.text(0.5, -0.08, "(b) gaussiano σ = 10", transform=e1.transAxes,
            ha="center", va="top", fontsize=11)
    e2.text(0.5, -0.08, "(c) ideal r = 10", transform=e2.transAxes,
            ha="center", va="top", fontsize=11)
    e3.text(0.5, -0.08, "(d) gaussiano σ = 60", transform=e3.transAxes,
            ha="center", va="top", fontsize=11)
    e4.text(0.5, -0.08, "(e) ideal r = 60", transform=e4.transAxes,
            ha="center", va="top", fontsize=11)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # O enunciado menciona D0 e sigma, mas a fórmula apresentada depende apenas
    # de sigma. Nesta implementação, D0 é usado como o mesmo parâmetro de
    # largura do filtro: D0 = sigma.
    d0_pequeno = 10
    d0_grande = 60
    sigma_pequeno = d0_pequeno
    sigma_grande = d0_grande

    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)

    mascara_gaussiana_10 = criar_mascara_gaussiana(original.shape, sigma_pequeno)
    mascara_gaussiana_60 = criar_mascara_gaussiana(original.shape, sigma_grande)
    mascara_ideal_10 = criar_mascara_passa_baixa_ideal(original.shape, d0_pequeno)
    mascara_ideal_60 = criar_mascara_passa_baixa_ideal(original.shape, d0_grande)

    gaussiana_10 = converter_para_uint8(aplicar_mascara_no_espectro(original, mascara_gaussiana_10))
    gaussiana_60 = converter_para_uint8(aplicar_mascara_no_espectro(original, mascara_gaussiana_60))
    ideal_10 = converter_para_uint8(aplicar_mascara_no_espectro(original, mascara_ideal_10))
    ideal_60 = converter_para_uint8(aplicar_mascara_no_espectro(original, mascara_ideal_60))

    salvar_comparacao(original, gaussiana_10, ideal_10, gaussiana_60,
                      ideal_60, ARQUIVO_SAIDA)
    print(f"Imagem usada: {ARQUIVO_ENTRADA.name}")
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")