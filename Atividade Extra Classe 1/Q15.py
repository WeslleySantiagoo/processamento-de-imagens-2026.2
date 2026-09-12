import numpy as np


def estatisticas(A):
    """Calcula somas e medias das linhas e colunas de uma matriz."""
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("A deve ser uma matriz")
    elif not (np.issubdtype(A.dtype, np.integer) or np.issubdtype(A.dtype, np.floating)):
        raise TypeError("A deve conter apenas numeros reais")

    soma_linhas = np.sum(A, axis=1)
    soma_colunas = np.sum(A, axis=0)
    media_linhas = soma_linhas / A.shape[1]
    media_colunas = soma_colunas / A.shape[0]

    return soma_linhas, soma_colunas, media_linhas, media_colunas


if __name__ == "__main__":
    n = 10
    # n = 5
    # n = 6
    # n = 2
    A = np.arange(n * n).reshape(n, n)

    soma_linhas, soma_colunas, media_linhas, media_colunas = estatisticas(A)

    print("A:\n", A)
    print("Soma das linhas:", soma_linhas, "Forma:", soma_linhas.shape)
    print("Soma das colunas:", soma_colunas, "Forma:", soma_colunas.shape)
    print("Media das linhas:", media_linhas, "Forma:", media_linhas.shape)
    print("Media das colunas:", media_colunas, "Forma:", media_colunas.shape)
