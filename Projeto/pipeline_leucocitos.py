from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def carregar_imagem_rgb(caminho):
    imagem_bgr = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem_bgr is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem em: {caminho}")
    return cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)


def rgb_para_hsi(imagem_rgb):
    """Converte imagem RGB para HSI de forma direta e vetorizada (LEC6 / Gonzalez & Woods).

    Retorna matiz [0, 360], saturacao [0, 1] e intensidade [0, 1].
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


def suavizar_e_realcar(imagem_rgb):
    """Suavização gaussiana para atenuação de ruído sem borrar bordas."""
    return cv2.GaussianBlur(imagem_rgb, (5, 5), 1.0)


def localizar_leucocitos_lamina(imagem_lamina_rgb, min_area=900, max_area=60000):
    """Varre a lâmina de campo amplo e detecta caixas delimitadoras de cada leucócito."""
    matiz, saturacao, intensidade = rgb_para_hsi(imagem_lamina_rgb)

    # Leucócitos possuem núcleo com afinidade tintorial púrpura/violeta
    # No HSI: matiz entre 220 e 340 graus, saturação relevante
    mascara_nucleos = (
        (matiz >= 220.0)
        & (matiz <= 340.0)
        & (saturacao >= 0.12)
        & (intensidade <= 0.88)
    ).astype(np.uint8) * 255

    # Operações morfológicas para agrupar lobos do mesmo núcleo e eliminar resíduos
    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_abertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mascara_fechada = cv2.morphologyEx(mascara_nucleos, cv2.MORPH_CLOSE, kernel_fechamento)
    mascara_limpa = cv2.morphologyEx(mascara_fechada, cv2.MORPH_OPEN, kernel_abertura)

    contornos, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    caixas_leucocitos = []
    altura_img, largura_img, _ = imagem_lamina_rgb.shape

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            pad = int(max(w, h) * 0.45)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(largura_img, x + w + pad)
            y2 = min(altura_img, y + h + pad)
            caixas_leucocitos.append((x1, y1, x2, y2))

    return caixas_leucocitos, mascara_limpa


def segmentar_nucleo_e_citoplasma(crop_celula_rgb):
    """Segmenta o núcleo e o citoplasma de um recorte de leucócito individual.

    Retorna a máscara binária do núcleo e a máscara binária do citoplasma.
    """
    matiz, saturacao, intensidade = rgb_para_hsi(crop_celula_rgb)

    # 1. Segmentação do núcleo por afinidade de matiz e intensidade
    mascara_nuc = (
        (matiz >= 210.0)
        & (matiz <= 345.0)
        & (saturacao >= 0.15)
        & (intensidade <= 0.80)
    ).astype(np.uint8) * 255

    kernel_nuc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_nuc = cv2.morphologyEx(mascara_nuc, cv2.MORPH_OPEN, kernel_nuc)
    mascara_nuc = cv2.morphologyEx(mascara_nuc, cv2.MORPH_CLOSE, kernel_nuc)

    # 2. Segmentação do corpo celular (célula inteira)
    mascara_celula = (
        (saturacao >= 0.08)
        & (intensidade <= 0.90)
    ).astype(np.uint8) * 255

    kernel_celula = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_celula = cv2.morphologyEx(mascara_celula, cv2.MORPH_CLOSE, kernel_celula)

    mascara_celula = cv2.bitwise_or(mascara_celula, mascara_nuc)
    mascara_citoplasma = cv2.bitwise_and(mascara_celula, cv2.bitwise_not(mascara_nuc))

    return mascara_nuc, mascara_citoplasma


def extrair_caracteristicas(crop_rgb, mascara_nucleo, mascara_citoplasma):
    """Extrai vetor numérico de descritores de forma, cor HSI e textura."""
    area_nucleo = float(np.count_nonzero(mascara_nucleo))
    area_citoplasma = float(np.count_nonzero(mascara_citoplasma))
    area_total = area_nucleo + area_citoplasma

    if area_total == 0:
        area_total = 1.0

    # 1. Razão Núcleo / Citoplasma
    razao_nc = area_nucleo / area_total

    # 2. Perímetro e Circularidade do núcleo
    contornos, _ = cv2.findContours(mascara_nucleo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contornos) > 0:
        c_maior = max(contornos, key=cv2.contourArea)
        perimetro_nucleo = float(cv2.arcLength(c_maior, True))
        num_lobos = float(len([c for c in contornos if cv2.contourArea(c) > 50]))
    else:
        perimetro_nucleo = 1.0
        num_lobos = 1.0

    if perimetro_nucleo > 0 and area_nucleo > 0:
        circularidade = float(4.0 * np.pi * area_nucleo / (perimetro_nucleo ** 2))
        compacidade = float((perimetro_nucleo ** 2) / area_nucleo)
    else:
        circularidade = 0.0
        compacidade = 0.0

    # 3. Momentos de Hu do núcleo (invariantes a escala e rotação)
    momentos = cv2.moments(mascara_nucleo)
    hu = cv2.HuMoments(momentos).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # 4. Estatísticas de Cor no HSI do citoplasma
    matiz, saturacao, intensidade = rgb_para_hsi(crop_rgb)
    idx_cito = mascara_citoplasma > 0

    if np.any(idx_cito):
        h_cito_media = float(np.mean(matiz[idx_cito]))
        s_cito_media = float(np.mean(saturacao[idx_cito]))
        i_cito_media = float(np.mean(intensidade[idx_cito]))
        i_cito_std = float(np.std(intensidade[idx_cito]))
    else:
        h_cito_media = 0.0
        s_cito_media = 0.0
        i_cito_media = 0.0
        i_cito_std = 0.0

    # 5. Suavidade / Textura de 1ª ordem do citoplasma
    suavidade = float(1.0 - (1.0 / (1.0 + (i_cito_std ** 2))))

    caracteristicas = {
        "area_nucleo": area_nucleo,
        "area_citoplasma": area_citoplasma,
        "razao_nc": razao_nc,
        "perimetro_nucleo": perimetro_nucleo,
        "circularidade": circularidade,
        "compacidade": compacidade,
        "num_lobos": num_lobos,
        "hu_1": float(hu_log[0]),
        "hu_2": float(hu_log[1]),
        "h_citoplasma": h_cito_media,
        "s_citoplasma": s_cito_media,
        "i_citoplasma": i_cito_media,
        "desvio_citoplasma": i_cito_std,
        "suavidade_citoplasma": suavidade,
    }

    return caracteristicas


def salvar_caracteristicas_parquet(lista_dicionarios, caminho_parquet):
    df = pd.DataFrame(lista_dicionarios)
    caminho = Path(caminho_parquet)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(caminho), index=False)
    return df


def carregar_caracteristicas_parquet(caminho_parquet):
    return pd.read_parquet(str(caminho_parquet))


def classificar_distancia_minima(X_treino, y_treino, X_amostra):
    """Classificador por Distância Mínima de Protótipos (Média da classe)."""
    classes_unicas = np.unique(y_treino)
    prototipos = {}
    for c in classes_unicas:
        prototipos[c] = np.mean(X_treino[y_treino == c], axis=0)

    predicoes = []
    for x in X_amostra:
        distancias = {c: np.linalg.norm(x - prototipos[c]) for c in classes_unicas}
        predicoes.append(min(distancias, key=distancias.get))

    return np.array(predicoes)


def classificar_knn(X_treino, y_treino, X_amostra, k=5):
    """Classificador k-Vizinhos Mais Próximos (k-NN) vetorizado."""
    predicoes = []
    for x in X_amostra:
        distancias = np.linalg.norm(X_treino - x, axis=1)
        indices_k = np.argsort(distancias)[:k]
        classes_k = y_treino[indices_k]
        valores, contagens = np.unique(classes_k, return_counts=True)
        predicoes.append(valores[np.argmax(contagens)])

    return np.array(predicoes)


def gerar_laudo_diferencial(classes_preditas):
    """Gera contagem e percentual das 5 linhagens leucocitárias."""
    linhagens = ["Neutrófilo", "Linfócito", "Monócito", "Eosinófilo", "Basófilo"]
    total = len(classes_preditas)

    laudo = []
    for linhagem in linhagens:
        contagem = int(np.sum(classes_preditas == linhagem))
        percentual = (contagem / total * 100.0) if total > 0 else 0.0
        laudo.append({
            "Linhagem": linhagem,
            "Contagem": contagem,
            "Percentual (%)": round(percentual, 1)
        })

    return pd.DataFrame(laudo), total
