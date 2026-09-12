import numpy as np


def operacoes(A):
    """Calcula soma, media e maximo em cada eixo de um array 3D."""
    A = np.asarray(A)

    if A.ndim != 3:
        raise ValueError("A deve ter tres dimensoes")
    elif A.size == 0:
        raise ValueError("A nao pode estar vazio")
    elif not (np.issubdtype(A.dtype, np.integer) or np.issubdtype(A.dtype, np.floating)):
        raise TypeError("A deve conter apenas numeros reais")

    soma_0 = np.sum(A, axis=0)
    soma_1 = np.sum(A, axis=1)
    soma_2 = np.sum(A, axis=2)

    media_0 = soma_0 / A.shape[0]
    media_1 = soma_1 / A.shape[1]
    media_2 = soma_2 / A.shape[2]

    maior_0 = np.max(A, axis=0)
    maior_1 = np.max(A, axis=1)
    maior_2 = np.max(A, axis=2)

    maior_global = np.max(A)
    i, j, k = np.where(A == maior_global)
    posicoes = np.stack((i, j, k), axis=1)

    return soma_0, soma_1, soma_2, media_0, media_1, media_2, maior_0, maior_1, maior_2, posicoes


if __name__ == "__main__":
    linhas = 4
    colunas = 5
    canais = 3
    # linhas = 2
    # colunas = 3
    # canais = 2
    A = np.arange(linhas * colunas * canais).reshape(linhas, colunas, canais)

    soma_0, soma_1, soma_2, media_0, media_1, media_2, maior_0, maior_1, maior_2, posicoes = operacoes(A)

    print("Forma de A:", A.shape)
    print("axis=0 remove a primeira dimensao")
    print("Soma axis=0:\n", soma_0, "\nForma:", soma_0.shape)
    print("Media axis=0:\n", media_0, "\nForma:", media_0.shape)
    print("Maior axis=0:\n", maior_0, "\nForma:", maior_0.shape)
    print("\naxis=1 remove a segunda dimensao")
    print("Soma axis=1:\n", soma_1, "\nForma:", soma_1.shape)
    print("Media axis=1:\n", media_1, "\nForma:", media_1.shape)
    print("Maior axis=1:\n", maior_1, "\nForma:", maior_1.shape)
    print("\naxis=2 remove a terceira dimensao")
    print("Soma axis=2:\n", soma_2, "\nForma:", soma_2.shape)
    print("Media axis=2:\n", media_2, "\nForma:", media_2.shape)
    print("Maior axis=2:\n", maior_2, "\nForma:", maior_2.shape)
    print("\nPosicao(oes) do maior valor global:", posicoes)
