from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens" / "monalisa.png"
ARQUIVO_SAIDA_PARTE_A = PASTA_ATUAL / "resultados" / "Q3_ParteA_original.png"
ARQUIVO_SAIDA_PARTE_B = PASTA_ATUAL / "resultados" / "Q3_ParteB_RGB.png"
ARQUIVO_SAIDA_PARTE_C = PASTA_ATUAL / "resultados" / "Q3_ParteC_HSI.png"
ARQUIVO_SAIDA_COMPARATIVO = PASTA_ATUAL / "resultados" / "Q3_ParteD_comparativo.png"


def carregar_imagem_rgb(caminho):
    imagem_bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem_bgr is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {caminho}")
    return cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)


def calcular_histograma(canal):
    """Calcula o histograma de frequências [0, 255] de forma direta."""
    histograma = np.zeros(256, dtype=np.int64)
    for valor in canal.ravel():
        histograma[int(valor)] += 1
    return histograma


def equalizar_canal(canal):
    """Aplica equalização de histograma manual usando a função de distribuição acumulada (CDF)."""
    histograma = calcular_histograma(canal)
    total_pixels = canal.size

    cdf = histograma.cumsum()
    cdf_min = cdf[cdf > 0].min()

    if total_pixels == cdf_min:
        return canal.copy()

    tabela_mapeamento = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255.0).astype(np.uint8)
    return tabela_mapeamento[canal]


def rgb_para_hsi(imagem_rgb):
    """Converte imagem RGB para componentes HSI vetorizadas conforme formulações teóricas."""
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


def hsi_para_rgb(matiz, saturacao, intensidade):
    """Converte componentes HSI de volta para RGB por setores circulares (RG, GB, BR)."""
    h = matiz % 360.0
    s = np.clip(saturacao, 0.0, 1.0)
    i = np.clip(intensidade, 0.0, 1.0)

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    mascara_rg = (h >= 0.0) & (h < 120.0)
    if np.any(mascara_rg):
        h_rad = np.radians(h[mascara_rg])
        b[mascara_rg] = i[mascara_rg] * (1.0 - s[mascara_rg])
        den = np.cos(np.radians(60.0) - h_rad)
        den = np.where(den == 0, 1e-6, den)
        r[mascara_rg] = i[mascara_rg] * (1.0 + (s[mascara_rg] * np.cos(h_rad)) / den)
        g[mascara_rg] = 3.0 * i[mascara_rg] - (r[mascara_rg] + b[mascara_rg])

    mascara_gb = (h >= 120.0) & (h < 240.0)
    if np.any(mascara_gb):
        h_shift = h[mascara_gb] - 120.0
        h_rad = np.radians(h_shift)
        r[mascara_gb] = i[mascara_gb] * (1.0 - s[mascara_gb])
        den = np.cos(np.radians(60.0) - h_rad)
        den = np.where(den == 0, 1e-6, den)
        g[mascara_gb] = i[mascara_gb] * (1.0 + (s[mascara_gb] * np.cos(h_rad)) / den)
        b[mascara_gb] = 3.0 * i[mascara_gb] - (r[mascara_gb] + g[mascara_gb])

    mascara_br = (h >= 240.0) & (h < 360.0)
    if np.any(mascara_br):
        h_shift = h[mascara_br] - 240.0
        h_rad = np.radians(h_shift)
        g[mascara_br] = i[mascara_br] * (1.0 - s[mascara_br])
        den = np.cos(np.radians(60.0) - h_rad)
        den = np.where(den == 0, 1e-6, den)
        b[mascara_br] = i[mascara_br] * (1.0 + (s[mascara_br] * np.cos(h_rad)) / den)
        r[mascara_br] = 3.0 * i[mascara_br] - (g[mascara_br] + b[mascara_br])

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def plotar_histograma_rgb(eixo, imagem_rgb, titulo):
    cores = ("red", "green", "blue")
    canais = ("R", "G", "B")
    for idx, (cor, nome) in enumerate(zip(cores, canais)):
        hist = calcular_histograma(imagem_rgb[:, :, idx])
        eixo.plot(hist, color=cor, label=f"Canal {nome}", linewidth=1.5)
    eixo.set_title(titulo)
    eixo.set_xlim([0, 255])
    eixo.set_xlabel("Intensidade")
    eixo.set_ylabel("Pixels")
    eixo.legend(loc="upper right")
    eixo.grid(alpha=0.3)


def gerar_painel_parte_a(imagem_original):
    # Funções de amostragem criadas com auxilio de IA para melhor visualização dos resultados
    figura, eixos = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    eixos[0].imshow(imagem_original)
    eixos[0].set_title("Parte A: Imagem Original")
    eixos[0].axis("off")

    plotar_histograma_rgb(eixos[1], imagem_original, "Histogramas dos Canais R, G e B (Original)")

    ARQUIVO_SAIDA_PARTE_A.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA_PARTE_A, dpi=180)
    plt.show()
    plt.close(figura)


def gerar_painel_parte_b(imagem_eq_rgb):
    # Funções de amostragem criadas com auxilio de IA para melhor visualização dos resultados
    figura, eixos = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    eixos[0].imshow(imagem_eq_rgb)
    eixos[0].set_title("Parte B: Equalização Independente em RGB")
    eixos[0].axis("off")

    plotar_histograma_rgb(eixos[1], imagem_eq_rgb, "Histogramas dos Canais R, G e B (Equalizados)")

    figura.savefig(ARQUIVO_SAIDA_PARTE_B, dpi=180)
    plt.show()
    plt.close(figura)


def gerar_painel_parte_c(intensidade_orig, hist_i_orig, hist_i_eq, imagem_eq_hsi):
    # Funções de amostragem criadas com auxilio de IA para melhor visualização dos resultados
    figura, eixos = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    eixos[0, 0].imshow(intensidade_orig, cmap="gray", vmin=0, vmax=255)
    eixos[0, 0].set_title("Componente Intensidade Original (I)")
    eixos[0, 0].axis("off")

    eixos[0, 1].plot(hist_i_orig, color="black", linewidth=1.5)
    eixos[0, 1].set_title("Histograma da Intensidade Original")
    eixos[0, 1].set_xlim([0, 255])
    eixos[0, 1].grid(alpha=0.3)

    eixos[1, 0].plot(hist_i_eq, color="black", linewidth=1.5)
    eixos[1, 0].set_title("Histograma da Intensidade Equalizada")
    eixos[1, 0].set_xlim([0, 255])
    eixos[1, 0].grid(alpha=0.3)

    eixos[1, 1].imshow(imagem_eq_hsi)
    eixos[1, 1].set_title("Imagem Final Reconstruída (Equalização HSI)")
    eixos[1, 1].axis("off")

    figura.savefig(ARQUIVO_SAIDA_PARTE_C, dpi=180)
    plt.show()
    plt.close(figura)


def gerar_painel_comparativo(img_original, img_eq_rgb, img_eq_hsi):
    # Funções de amostragem criadas com auxilio de IA para melhor visualização dos resultados
    figura, eixos = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    eixos[0].imshow(img_original)
    eixos[0].set_title("Original (Baixo Contraste)")
    eixos[0].axis("off")

    eixos[1].imshow(img_eq_rgb)
    eixos[1].set_title("Equalização RGB Independente")
    eixos[1].axis("off")

    eixos[2].imshow(img_eq_hsi)
    eixos[2].set_title("Equalização HSI (Canal I)")
    eixos[2].axis("off")

    figura.savefig(ARQUIVO_SAIDA_COMPARATIVO, dpi=180)
    plt.show()
    plt.close(figura)


def main():
    imagem_rgb = carregar_imagem_rgb(ARQUIVO_ENTRADA)
    print(f"Imagem carregada: {ARQUIVO_ENTRADA.name} | Resolução: {imagem_rgb.shape}")

    print("\n--- Parte A: Análise da Imagem Original ---")
    for i, canal in enumerate(("R", "G", "B")):
        vals = imagem_rgb[:, :, i]
        print(f"Canal {canal} -> Min: {vals.min()}, Max: {vals.max()}, Média: {vals.mean():.2f}, Desvio: {vals.std():.2f}")
    gerar_painel_parte_a(imagem_rgb)

    print("\n--- Parte B: Equalização Independente dos Canais RGB ---")
    r_eq = equalizar_canal(imagem_rgb[:, :, 0])
    g_eq = equalizar_canal(imagem_rgb[:, :, 1])
    b_eq = equalizar_canal(imagem_rgb[:, :, 2])
    imagem_eq_rgb = np.stack([r_eq, g_eq, b_eq], axis=2)
    gerar_painel_parte_b(imagem_eq_rgb)

    print("\n--- Parte C: Equalização no Espaço HSI ---")
    matiz, saturacao, intensidade = rgb_para_hsi(imagem_rgb)
    intensidade_uint8 = np.round(intensidade * 255.0).astype(np.uint8)
    hist_i_orig = calcular_histograma(intensidade_uint8)

    intensidade_eq_uint8 = equalizar_canal(intensidade_uint8)
    hist_i_eq = calcular_histograma(intensidade_eq_uint8)

    intensidade_eq_norm = intensidade_eq_uint8.astype(np.float64) / 255.0
    imagem_eq_hsi = hsi_para_rgb(matiz, saturacao, intensidade_eq_norm)
    gerar_painel_parte_c(intensidade_uint8, hist_i_orig, hist_i_eq, imagem_eq_hsi)

    print("\n--- Parte D: Comparação Geral dos Métodos ---")
    gerar_painel_comparativo(imagem_rgb, imagem_eq_rgb, imagem_eq_hsi)
    print("Processamento concluído com sucesso!")


if __name__ == "__main__":
    main()

