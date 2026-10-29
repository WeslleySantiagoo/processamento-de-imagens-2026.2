from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens" / "baboon_colorida.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q2_resultados.png"


def carregar_imagem_rgb(caminho):
    imagem_bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem_bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {caminho}")
    return cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)


def rgb_para_hsi(imagem_rgb):
    """Converte uma imagem RGB para o espaço de cores HSI de forma manual e vetorizada,

    seguindo estritamente as equações teóricas dos slides (70-73):

    1. R, G, B normalizados em [0, 1]
    2. I = (R + G + B) / 3
    3. S = 1 - (3 * min(R, G, B)) / (R + G + B), com S = 0 se (R + G + B) == 0
    4. theta = arccos( 0.5 * [(R - G) + (R - B)] / sqrt((R - G)^2 + (R - B)*(G - B)) )
       Se B > G: H = 360 - theta, senão H = theta (em graus [0, 360])
    """
    img_norm = imagem_rgb.astype(np.float64) / 255.0
    r = img_norm[:, :, 0]
    g = img_norm[:, :, 1]
    b = img_norm[:, :, 2]

    soma_rgb = r + g + b
    intensidade = soma_rgb / 3.0

    min_rgb = np.minimum(np.minimum(r, g), b)
    saturacao = np.zeros_like(intensidade)
    mascara_soma = soma_rgb > 1e-6
    saturacao[mascara_soma] = 1.0 - (3.0 * min_rgb[mascara_soma] / soma_rgb[mascara_soma])

    numerador = 0.5 * ((r - g) + (r - b))
    denominador = np.sqrt((r - g) ** 2 + (r - b) * (g - b))

    theta = np.zeros_like(intensidade)
    mascara_den = denominador > 1e-6

    razao = np.clip(numerador[mascara_den] / denominador[mascara_den], -1.0, 1.0)
    theta[mascara_den] = np.arccos(razao)

    matiz = np.degrees(theta)

    mascara_b_maior = b > g
    matiz[mascara_b_maior] = 360.0 - matiz[mascara_b_maior]

    matiz[saturacao == 0] = 0.0

    return matiz, saturacao, intensidade


def isolar_objeto_azul(imagem_rgb, matiz, saturacao, intensidade):
    """Cria máscara binária para segmentar a região azul (ex.: focinho do mandril/baboon)

    no espaço HSI e isola o objeto sobre fundo preto.
    """
    mascara = (
        (matiz >= 180.0)
        & (matiz <= 260.0)
        & (saturacao >= 0.15)
        & (intensidade >= 0.10)
    )

    mascara_binaria = (mascara.astype(np.uint8)) * 255

    objeto_isolado = np.zeros_like(imagem_rgb)
    objeto_isolado[mascara] = imagem_rgb[mascara]

    return mascara_binaria, objeto_isolado


def criar_painel_visual(img_original, matiz, saturacao, intensidade, mascara, isolado):
    # Função criada com auxilio de IA, apenas para amostragem
    figura, eixos = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    eixos[0, 0].imshow(img_original)
    eixos[0, 0].set_title("Imagem Original (RGB)")
    eixos[0, 0].axis("off")

    im_h = eixos[0, 1].imshow(matiz, cmap="hsv", vmin=0, vmax=360)
    eixos[0, 1].set_title("Componente Matiz (H) [0° a 360°]")
    eixos[0, 1].axis("off")
    figura.colorbar(im_h, ax=eixos[0, 1], fraction=0.046, pad=0.04)

    im_s = eixos[0, 2].imshow(saturacao, cmap="gray", vmin=0, vmax=1)
    eixos[0, 2].set_title("Componente Saturação (S) [0 a 1]")
    eixos[0, 2].axis("off")
    figura.colorbar(im_s, ax=eixos[0, 2], fraction=0.046, pad=0.04)

    im_i = eixos[1, 0].imshow(intensidade, cmap="gray", vmin=0, vmax=1)
    eixos[1, 0].set_title("Componente Intensidade (I) [0 a 1]")
    eixos[1, 0].axis("off")
    figura.colorbar(im_i, ax=eixos[1, 0], fraction=0.046, pad=0.04)

    eixos[1, 1].imshow(mascara, cmap="gray")
    eixos[1, 1].set_title("Máscara Binária (Azul Isolado)")
    eixos[1, 1].axis("off")

    eixos[1, 2].imshow(isolado)
    eixos[1, 2].set_title("Objeto Segmentado em Cores")
    eixos[1, 2].axis("off")

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=180)
    plt.show()
    plt.close(figura)
    print(f"Visualização da Q2 salva com sucsso em: {ARQUIVO_SAIDA}")


def main():
    imagem_rgb = carregar_imagem_rgb(ARQUIVO_ENTRADA)
    print(f"Imagem carregada: {ARQUIVO_ENTRADA.name} | Resolução: {imagem_rgb.shape}")

    matiz, saturacao, intensidade = rgb_para_hsi(imagem_rgb)

    print(f"Matiz (H) -> min: {matiz.min():.2f}°, max: {matiz.max():.2f}°, média: {matiz.mean():.2f}°")
    print(f"Saturação (S) -> min: {saturacao.min():.2f}, max: {saturacao.max():.2f}, média: {saturacao.mean():.2f}")
    print(f"Intensidade (I) -> min: {intensidade.min():.2f}, max: {intensidade.max():.2f}, média: {intensidade.mean():.2f}")

    mascara_binaria, objeto_isolado = isolar_objeto_azul(imagem_rgb, matiz, saturacao, intensidade)
    pixels_segmentados = np.count_nonzero(mascara_binaria)
    porcentagem = (pixels_segmentados / mascara_binaria.size) * 100
    print(f"Pixels do objeto azul segmentados: {pixels_segmentados} ({porcentagem:.2f}% da imagem)")

    criar_painel_visual(imagem_rgb, matiz, saturacao, intensidade, mascara_binaria, objeto_isolado)


if __name__ == "__main__":
    main()

