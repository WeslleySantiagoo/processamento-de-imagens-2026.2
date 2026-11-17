from pathlib import Path
import urllib.request
import zipfile
import shutil
import cv2
import numpy as np


PASTA_DADOS = Path(__file__).resolve().parent / "dados"
PASTA_LAMINAS = PASTA_DADOS / "laminas_completas"
PASTA_CELULAS = PASTA_DADOS / "celulas_treino"

URL_BCCD = "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/"
URL_BLOOD_CELLS = "https://raw.githubusercontent.com/tahasen/Blood-Cells-Classification-using-Deep-Learning/master/dataset-master/JPEGImages/"


def baixar_imagem(url, caminho_destino):
    caminho_destino = Path(caminho_destino)
    caminho_destino.parent.mkdir(parents=True, exist_ok=True)
    if not caminho_destino.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resposta, open(caminho_destino, "wb") as f:
            f.write(resposta.read())


def preparar_laminas_bccd():
    print("Baixando lâminas microscópicas completas do BCCD...")
    laminas = [
        "BloodImage_00000.jpg",
        "BloodImage_00005.jpg",
        "BloodImage_00022.jpg",
        "BloodImage_00035.jpg",
        "BloodImage_00062.jpg",
    ]
    for nome in laminas:
        destino = PASTA_LAMINAS / nome
        try:
            baixar_imagem(URL_BCCD + nome, destino)
            print(f"  [OK] {nome}")
        except Exception as e:
            print(f"  [AVISO] Falha ao baixar {nome}: {e}")


def gerar_amostras_sinteticas_se_necessario():
    """Garante amostras de referência para as 5 linhagens para garantir execução autônoma."""
    classes = {
        "neutrofilo": {"cor_citoplasma": (195, 175, 205), "lobos": 3, "razao_nc": 0.55, "granulos": True},
        "linfocito": {"cor_citoplasma": (175, 190, 225), "lobos": 1, "razao_nc": 0.85, "granulos": False},
        "monocito": {"cor_citoplasma": (185, 195, 215), "lobos": 1, "razao_nc": 0.45, "granulos": False},
        "eosinofilo": {"cor_citoplasma": (160, 150, 230), "lobos": 2, "razao_nc": 0.50, "granulos": True},
        "basofilo": {"cor_citoplasma": (190, 140, 160), "lobos": 2, "razao_nc": 0.65, "granulos": True},
    }

    # Para cada classe, cria 10 amostras calibradas se ainda não existirem
    for classe, params in classes.items():
        pasta_classe = PASTA_CELULAS / classe
        pasta_classe.mkdir(parents=True, exist_ok=True)
        existentes = list(pasta_classe.glob("*.jpg")) + list(pasta_classe.glob("*.png"))
        if len(existentes) >= 8:
            continue

        print(f"Gerando amostras padrão para calibração: {classe}...")
        for i in range(12):
            img = np.full((120, 120, 3), (235, 230, 240), dtype=np.uint8)  # Fundo do esfregaço
            # Adiciona hemácias de fundo
            cv2.circle(img, (20, 25), 18, (180, 175, 225), -1)
            cv2.circle(img, (100, 105), 16, (185, 170, 230), -1)

            centro = (60 + np.random.randint(-3, 4), 60 + np.random.randint(-3, 4))
            raio_celula = int(40 + np.random.randint(-2, 3))
            
            # Citoplasma
            cor_cito = params["cor_citoplasma"]
            cv2.circle(img, centro, raio_celula, cor_cito, -1)

            # Núcleo (púrpura escuro característico)
            cor_nucleo = (115 + np.random.randint(-10, 10), 45 + np.random.randint(-5, 5), 85 + np.random.randint(-10, 10))
            if params["lobos"] == 1:
                raio_nuc = int(raio_celula * np.sqrt(params["razao_nc"]))
                if classe == "monocito":
                    # Formato reniforme/feijão
                    cv2.ellipse(img, centro, (raio_nuc, int(raio_nuc * 0.7)), 30, 0, 360, cor_nucleo, -1)
                    cv2.circle(img, (centro[0] + 5, centro[1]), int(raio_nuc * 0.35), cor_cito, -1)
                else:
                    cv2.circle(img, centro, raio_nuc, cor_nucleo, -1)
            elif params["lobos"] == 2:
                raio_nuc = int(raio_celula * np.sqrt(params["razao_nc"]) * 0.7)
                cv2.circle(img, (centro[0] - 12, centro[1]), raio_nuc, cor_nucleo, -1)
                cv2.circle(img, (centro[0] + 12, centro[1]), raio_nuc, cor_nucleo, -1)
                cv2.line(img, (centro[0] - 12, centro[1]), (centro[0] + 12, centro[1]), cor_nucleo, 4)
            else:
                # 3 lobos (Neutrófilo)
                raio_nuc = int(raio_celula * np.sqrt(params["razao_nc"]) * 0.55)
                cv2.circle(img, (centro[0] - 12, centro[1] - 8), raio_nuc, cor_nucleo, -1)
                cv2.circle(img, (centro[0] + 12, centro[1] - 8), raio_nuc, cor_nucleo, -1)
                cv2.circle(img, (centro[0], centro[1] + 14), raio_nuc, cor_nucleo, -1)
                cv2.line(img, (centro[0] - 12, centro[1] - 8), (centro[0] + 12, centro[1] - 8), cor_nucleo, 3)
                cv2.line(img, (centro[0] + 12, centro[1] - 8), (centro[0], centro[1] + 14), cor_nucleo, 3)

            # Granulações citoplasmáticas
            if params["granulos"]:
                cor_granulo = (140, 30, 200) if classe == "basofilo" else (80, 70, 220)
                for _ in range(35):
                    gx = centro[0] + np.random.randint(-raio_celula + 5, raio_celula - 5)
                    gy = centro[1] + np.random.randint(-raio_celula + 5, raio_celula - 5)
                    if (gx - centro[0])**2 + (gy - centro[1])**2 < (raio_celula - 4)**2:
                        img[gy, gx] = cor_granulo

            # Ruído gaussiano leve
            ruido = np.random.normal(0, 3, img.shape).astype(np.int16)
            img_final = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

            caminho_arquivo = pasta_classe / f"{classe}_{i+1:02d}.png"
            cv2.imwrite(str(caminho_arquivo), cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    preparar_laminas_bccd()
    gerar_amostras_sinteticas_se_necessario()
    print("Preparação de dados concluída!")
