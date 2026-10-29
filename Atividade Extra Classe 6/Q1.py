from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens" / "watch.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q1_resultados.png"


def carregar_imagem_rgb(caminho):
    imagem_bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem_bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {caminho}")
    return cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)


def aplicar_transformacao_sepia(imagem_rgb):
    """Aplica a transformação de cor sépia definida em Q1 (a) de forma vetorizada:

    R' = 0.393*R + 0.769*G + 0.189*B
    G' = 0.349*R + 0.686*G + 0.168*B
    B' = 0.272*R + 0.534*G + 0.131*B
    Valores superiores a 255 são limitados em 255.
    """
    img_float = imagem_rgb.astype(np.float64)
    r = img_float[:, :, 0]
    g = img_float[:, :, 1]
    b = img_float[:, :, 2]

    r_novo = 0.393 * r + 0.769 * g + 0.189 * b
    g_novo = 0.349 * r + 0.686 * g + 0.168 * b
    b_novo = 0.272 * r + 0.534 * g + 0.131 * b

    # Limita os valores no intervalo [0, 255]
    r_clip = np.clip(r_novo, 0, 255)
    g_clip = np.clip(g_novo, 0, 255)
    b_clip = np.clip(b_novo, 0, 255)

    imagem_sepia = np.stack([r_clip, g_clip, b_clip], axis=2).astype(np.uint8)
    return imagem_sepia


def converter_para_cinza_ponderado(imagem_rgb):
    """Converte a imagem RGB para uma única banda de intensidade conforme Q1 (b):

    I = 0.2989*R + 0.5870*G + 0.1140*B
    """
    img_float = imagem_rgb.astype(np.float64)
    r = img_float[:, :, 0]
    g = img_float[:, :, 1]
    b = img_float[:, :, 2]

    intensidade = 0.2989 * r + 0.5870 * g + 0.1140 * b
    intensidade_clip = np.clip(intensidade, 0, 255)

    return intensidade_clip.astype(np.uint8)


def criar_painel_comparativo(img_original, img_sepia, img_cinza):
    figura, eixos = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    eixos[0].imshow(img_original)
    eixos[0].set_title("Imagem Original (RGB)")
    eixos[0].axis("off")

    eixos[1].imshow(img_sepia)
    eixos[1].set_title("Q1(a) - Transformação Sépia")
    eixos[1].axis("off")

    eixos[2].imshow(img_cinza, cmap="gray", vmin=0, vmax=255)
    eixos[2].set_title("Q1(b) - Escala de Cinza Ponderada")
    eixos[2].axis("off")

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=180)
    plt.show()
    print(f"Resultado visual salvo com sucesso em: {ARQUIVO_SAIDA}")


def main():
    imagem_rgb = carregar_imagem_rgb(ARQUIVO_ENTRADA)
    print(f"Imagem carregada: {ARQUIVO_ENTRADA.name} com formato {imagem_rgb.shape}")

    imagem_sepia = aplicar_transformacao_sepia(imagem_rgb)
    imagem_cinza = converter_para_cinza_ponderado(imagem_rgb)

    print(f"Sépia -> Min: {imagem_sepia.min()}, Max: {imagem_sepia.max()}, Médias por canal (R, G, B): "
          f"({imagem_sepia[:, :, 0].mean():.2f}, {imagem_sepia[:, :, 1].mean():.2f}, {imagem_sepia[:, :, 2].mean():.2f})")
    print(f"Cinza -> Min: {imagem_cinza.min()}, Max: {imagem_cinza.max()}, Média de intensidade: {imagem_cinza.mean():.2f}")

    criar_painel_comparativo(imagem_rgb, imagem_sepia, imagem_cinza)


if __name__ == "__main__":
    main()

