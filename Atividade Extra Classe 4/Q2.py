"""Q2 — Reconstrução de imagem a partir da Transformada de Fourier 2D."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "butterfly.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q2_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega uma imagem como matriz de intensidades em tons de cinza."""
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)

    if imagem is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {caminho}")

    return imagem


def normalizar_para_visualizacao(imagem: np.ndarray):
    """Converte uma matriz numérica para o intervalo visual de 0 a 255."""
    minimo = imagem.min()
    maximo = imagem.max()

    if maximo == minimo:
        return np.zeros(imagem.shape, dtype=np.uint8)

    imagem_normalizada = (imagem - minimo) * (255 / (maximo - minimo))
    return np.rint(imagem_normalizada).astype(np.uint8)


def reconstruir_por_espectro(espectro: np.ndarray):
    """Reconstrói uma imagem real usando a transformada inversa de Fourier."""
    imagem_complexa = np.fft.ifft2(espectro)
    return np.real(imagem_complexa)


def calcular_reconstrucoes(imagem: np.ndarray):
    """Calcula as reconstruções completa, por magnitude e por fase isoladas."""
    imagem_float = imagem.astype(np.float32)
    espectro = np.fft.fft2(imagem_float)
    magnitude = np.abs(espectro)
    fase = np.angle(espectro)

    reconstruida = reconstruir_por_espectro(espectro)

    fase_zero = np.zeros(fase.shape)
    espectro_apenas_magnitude = magnitude * np.exp(1j * fase_zero)
    apenas_magnitude = reconstruir_por_espectro(espectro_apenas_magnitude)

    magnitude_constante = np.ones(magnitude.shape)
    espectro_apenas_fase = magnitude_constante * np.exp(1j * fase)
    apenas_fase = reconstruir_por_espectro(espectro_apenas_fase)

    diferenca_absoluta = np.abs(imagem_float - reconstruida)
    erro_maximo = np.max(diferenca_absoluta)
    erro_quadratico_medio = np.mean(diferenca_absoluta ** 2)

    return reconstruida, diferenca_absoluta, apenas_magnitude, apenas_fase, erro_maximo, erro_quadratico_medio


def salvar_comparacao(original: np.ndarray, reconstruida: np.ndarray,
                      diferenca: np.ndarray, apenas_magnitude: np.ndarray,
                      apenas_fase: np.ndarray, saida: Path):
    """Exibe e salva as reconstruções pedidas no enunciado."""
    # Usado IA apenas nessa função apenas para amostragem
    saida.parent.mkdir(parents=True, exist_ok=True)

    figura, eixos = plt.subplots(2, 3, figsize=(13, 8))
    figura.subplots_adjust(wspace=0.04, hspace=0.22, bottom=0.08)
    e0, e1, e2, e3, e4, e5 = eixos.flat

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(reconstruida, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(diferenca, cmap="gray"); e2.axis("off")
    e3.imshow(normalizar_para_visualizacao(apenas_magnitude), cmap="gray", vmin=0, vmax=255); e3.axis("off")
    e4.imshow(normalizar_para_visualizacao(apenas_fase), cmap="gray", vmin=0, vmax=255); e4.axis("off")
    e5.axis("off")

    e0.text(0.5, -0.08, "(a) imagem original", transform=e0.transAxes,
            ha="center", va="top", fontsize=11)
    e1.text(0.5, -0.08, "(b) reconstrução completa", transform=e1.transAxes,
            ha="center", va="top", fontsize=11)
    e2.text(0.5, -0.08, "(c) diferença absoluta", transform=e2.transAxes,
            ha="center", va="top", fontsize=11)
    e3.text(0.5, -0.08, "(d) somente magnitude", transform=e3.transAxes,
            ha="center", va="top", fontsize=11)
    e4.text(0.5, -0.08, "(e) somente fase", transform=e4.transAxes,
            ha="center", va="top", fontsize=11)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)
    reconstruida, diferenca, apenas_magnitude, apenas_fase, erro_maximo, erro_quadratico_medio = calcular_reconstrucoes(original)

    salvar_comparacao(original, reconstruida, diferenca, apenas_magnitude,
                      apenas_fase, ARQUIVO_SAIDA)

    print(f"Imagem usada: {ARQUIVO_ENTRADA.name}")
    print(f"Erro máximo da reconstrução: {erro_maximo:.10f}")
    print(f"Erro quadrático médio da reconstrução: {erro_quadratico_medio:.10f}")
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
