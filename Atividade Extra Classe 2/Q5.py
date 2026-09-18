"""Q5 — Construção de mosaico 4 * 4 a partir de uma imagem monocromática."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PASTA_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = PASTA_ATUAL / "imagens de entrada" / "baboon_monocromatica.png"
ARQUIVO_SAIDA = PASTA_ATUAL / "resultados" / "Q5_resultados.png"


def carregar_imagem_monocromatica(caminho: Path):
    """Carrega a imagem como matriz de intensidades inteiras entre 0 e 255."""
    imagem = plt.imread(caminho)

    if np.issubdtype(imagem.dtype, np.floating):
        imagem = np.rint(imagem * 255).astype(np.uint8)

    if imagem.ndim == 3:
        imagem = imagem[..., 0]

    return imagem


def construir_mosaico(imagem: np.ndarray, nova_ordem: np.ndarray,
                      quantidade_blocos_por_lado: int):
    """Reorganiza os blocos da imagem de acordo com a matriz nova_ordem."""
    altura, largura = imagem.shape

    if quantidade_blocos_por_lado <= 0:
        raise ValueError("A quantidade de blocos por lado deve ser maior que zero.")

    if altura % quantidade_blocos_por_lado != 0 or largura % quantidade_blocos_por_lado != 0:
        raise ValueError("A altura e a largura devem ser múltiplas da quantidade de blocos.")

    ordem_esperada = np.arange(1, quantidade_blocos_por_lado ** 2 + 1)
    ordem_informada = np.sort(nova_ordem.ravel())

    if nova_ordem.shape != (quantidade_blocos_por_lado, quantidade_blocos_por_lado):
        raise ValueError("A nova ordem deve ter o mesmo tamanho da grade de blocos.")

    if not np.array_equal(ordem_informada, ordem_esperada):
        raise ValueError("A nova ordem deve conter cada número de bloco uma única vez.")

    altura_bloco = altura // quantidade_blocos_por_lado
    largura_bloco = largura // quantidade_blocos_por_lado

    # Separa a imagem em uma grade de blocos de X por X.
    blocos_em_grade = imagem.reshape(quantidade_blocos_por_lado, altura_bloco,
                                     quantidade_blocos_por_lado, largura_bloco)
    blocos_em_grade = blocos_em_grade.transpose(0, 2, 1, 3)

    # Numera os blocos naturalmente e seleciona-os na nova ordem.
    blocos_naturais = blocos_em_grade.reshape(quantidade_blocos_por_lado ** 2,
                                              altura_bloco, largura_bloco)
    indices_nova_ordem = nova_ordem.ravel() - 1
    blocos_reorganizados = blocos_naturais[indices_nova_ordem]

    # Reagrupa a grade de blocos na matriz final da imagem mosaico.
    mosaico_em_grade = blocos_reorganizados.reshape(quantidade_blocos_por_lado,
                                                     quantidade_blocos_por_lado,
                                                     altura_bloco,
                                                     largura_bloco)
    mosaico = mosaico_em_grade.transpose(0, 2, 1, 3).reshape(altura, largura)

    return mosaico


def gerar_nova_ordem(quantidade_blocos_por_lado: int):
    """Gera uma ordem aleatória, sem repetição, para uma grade X * X."""
    if quantidade_blocos_por_lado <= 0:
        raise ValueError("A quantidade de blocos por lado deve ser maior que zero.")

    quantidade_total_blocos = quantidade_blocos_por_lado ** 2
    numeros_dos_blocos = np.arange(1, quantidade_total_blocos + 1)
    numeros_aleatorios = np.random.permutation(numeros_dos_blocos)

    return numeros_aleatorios.reshape(quantidade_blocos_por_lado,
                                     quantidade_blocos_por_lado)


def adicionar_tabela(eixo, ordem: np.ndarray):
    """Desenha uma tabela para representar a ordem dos blocos."""
    eixo.axis("off")
    tabela = eixo.table(cellText=ordem.astype(str), cellLoc="center",
                         loc="center", bbox=[0.06, 0.14, 0.88, 0.80])
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(13)


def salvar_comparacao(original: np.ndarray, ordem_original: np.ndarray,
                      nova_ordem: np.ndarray, mosaico: np.ndarray,
                      quantidade_blocos_por_lado: int, saida: Path):
    """Exibe e salva a comparação entre os blocos originais e o mosaico."""
    # Usado IA apenas para amostragem visual ficar aparentavel e semelhante ao exemplo da atividade
    figura, eixos = plt.subplots(1, 4, figsize=(14, 4.4))
    figura.subplots_adjust(wspace=0.08, bottom=0.16)

    e0, e1, e2, e3 = eixos

    e0.imshow(original, cmap="gray", vmin=0, vmax=255); e0.axis("off")
    adicionar_tabela(e1, ordem_original)
    adicionar_tabela(e2, nova_ordem)
    e3.imshow(mosaico, cmap="gray", vmin=0, vmax=255); e3.axis("off")

    e0.text(0.5, -0.08, "(a) imagem", transform=e0.transAxes,
            ha="center", va="top", fontsize=12)
    e1.text(0.5, 0.03, f"(b) ordem dos blocos ({quantidade_blocos_por_lado} * {quantidade_blocos_por_lado})", transform=e1.transAxes,
            ha="center", va="top", fontsize=12)
    e2.text(0.5, 0.03, f"(c) nova ordem dos blocos ({quantidade_blocos_por_lado} * {quantidade_blocos_por_lado})", transform=e2.transAxes,
            ha="center", va="top", fontsize=12)
    e3.text(0.5, -0.08, "(d) mosaico", transform=e3.transAxes,
            ha="center", va="top", fontsize=12)

    figura.savefig(saida, dpi=180, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    quantidade_blocos_por_lado = 4
    ordem_original = np.arange(1, quantidade_blocos_por_lado ** 2 + 1)
    ordem_original = ordem_original.reshape(quantidade_blocos_por_lado,
                                            quantidade_blocos_por_lado)
    nova_ordem = gerar_nova_ordem(quantidade_blocos_por_lado)

    original = carregar_imagem_monocromatica(ARQUIVO_ENTRADA)
    mosaico = construir_mosaico(original, nova_ordem, quantidade_blocos_por_lado)

    salvar_comparacao(original, ordem_original, nova_ordem, mosaico,
                      quantidade_blocos_por_lado, ARQUIVO_SAIDA)
    print(f"Imagem comparativa salva em: {ARQUIVO_SAIDA}")
