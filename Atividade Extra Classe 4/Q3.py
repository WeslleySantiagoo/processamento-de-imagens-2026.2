"""Q3 — Importância da fase na reconstrução de imagens."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_IMAGEM_A = PASTA_ATUAL / "imagens de entrada" / "baboon_monocromatica.png"
ARQUIVO_IMAGEM_B = PASTA_ATUAL / "imagens de entrada" / "barbara.pgm"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q3_resultados.png"


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
    """Reconstrói uma imagem real pela transformada inversa de Fourier."""
    return np.real(np.fft.ifft2(espectro))


def combinar_magnitude_e_fase(imagem_magnitude: np.ndarray,
                              imagem_fase: np.ndarray):
    """Reconstrói usando magnitude de uma imagem e fase de outra."""
    if imagem_magnitude.shape != imagem_fase.shape:
        raise ValueError("As imagens devem possuir as mesmas dimensões.")

    espectro_magnitude = np.fft.fft2(imagem_magnitude.astype(np.float32))
    espectro_fase = np.fft.fft2(imagem_fase.astype(np.float32))

    magnitude = np.abs(espectro_magnitude)
    fase = np.angle(espectro_fase)
    espectro_hibrido = magnitude * np.exp(1j * fase)

    return reconstruir_por_espectro(espectro_hibrido)


def reconstruir_apenas_magnitude(imagem: np.ndarray):
    """Reconstrói a imagem mantendo a magnitude e definindo a fase como zero."""
    espectro = np.fft.fft2(imagem.astype(np.float32))
    magnitude = np.abs(espectro)
    fase_zero = np.zeros(magnitude.shape)

    return reconstruir_por_espectro(magnitude * np.exp(1j * fase_zero))


def reconstruir_apenas_fase(imagem: np.ndarray):
    """Reconstrói a imagem com fase original e magnitude constante igual a um."""
    espectro = np.fft.fft2(imagem.astype(np.float32))
    fase = np.angle(espectro)
    magnitude_constante = np.ones(fase.shape)

    return reconstruir_por_espectro(magnitude_constante * np.exp(1j * fase))


def salvar_comparacao(imagem_a: np.ndarray, imagem_b: np.ndarray,
                      magnitude_a_fase_b: np.ndarray,
                      magnitude_b_fase_a: np.ndarray,
                      apenas_magnitude: np.ndarray, apenas_fase: np.ndarray,
                      saida: Path):
    """Exibe e salva as imagens originais e as reconstruções comparativas."""
    # Usado IA apenas para melhor amostragem
    saida.parent.mkdir(parents=True, exist_ok=True)

    figura, eixos = plt.subplots(2, 3, figsize=(12, 8))
    figura.subplots_adjust(wspace=0.04, hspace=0.20, bottom=0.08)
    e0, e1, e2, e3, e4, e5 = eixos.flat

    e0.imshow(imagem_a, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    e1.imshow(imagem_b, cmap="gray", vmin=0, vmax=255); e1.axis("off")
    e2.imshow(normalizar_para_visualizacao(magnitude_a_fase_b), cmap="gray", vmin=0, vmax=255); e2.axis("off")
    e3.imshow(normalizar_para_visualizacao(magnitude_b_fase_a), cmap="gray", vmin=0, vmax=255); e3.axis("off")
    e4.imshow(normalizar_para_visualizacao(apenas_magnitude), cmap="gray", vmin=0, vmax=255); e4.axis("off")
    e5.imshow(normalizar_para_visualizacao(apenas_fase), cmap="gray", vmin=0, vmax=255); e5.axis("off")

    e0.text(0.5, -0.08, "(a) imagem A: baboon", transform=e0.transAxes,
            ha="center", va="top", fontsize=10)
    e1.text(0.5, -0.08, "(b) imagem B: barbara", transform=e1.transAxes,
            ha="center", va="top", fontsize=10)
    e2.text(0.5, -0.08, "(c) magnitude A + fase B", transform=e2.transAxes,
            ha="center", va="top", fontsize=10)
    e3.text(0.5, -0.08, "(d) magnitude B + fase A", transform=e3.transAxes,
            ha="center", va="top", fontsize=10)
    e4.text(0.5, -0.08, "(e) somente magnitude A", transform=e4.transAxes,
            ha="center", va="top", fontsize=10)
    e5.text(0.5, -0.08, "(f) somente fase A", transform=e5.transAxes,
            ha="center", va="top", fontsize=10)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    imagem_a = carregar_imagem_monocromatica(ARQUIVO_IMAGEM_A)
    imagem_b = carregar_imagem_monocromatica(ARQUIVO_IMAGEM_B)

    magnitude_a_fase_b = combinar_magnitude_e_fase(imagem_a, imagem_b)
    magnitude_b_fase_a = combinar_magnitude_e_fase(imagem_b, imagem_a)
    apenas_magnitude = reconstruir_apenas_magnitude(imagem_a)
    apenas_fase = reconstruir_apenas_fase(imagem_a)

    salvar_comparacao(imagem_a, imagem_b, magnitude_a_fase_b,
                      magnitude_b_fase_a, apenas_magnitude, apenas_fase,
                      ARQUIVO_SAIDA)
    print(f"Imagem A usada: {ARQUIVO_IMAGEM_A.name}")
    print(f"Imagem B usada: {ARQUIVO_IMAGEM_B.name}")
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
