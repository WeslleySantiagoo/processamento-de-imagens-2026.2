import numpy as np


def media_movel(x, n):
    """Calcula a media movel de um vetor com janelas de tamanho n."""
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x deve ser um vetor unidimensional")
    elif not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)):
        raise TypeError("x deve conter apenas numeros reais")
    elif not isinstance(n, (int, np.integer)):
        raise TypeError("n deve ser um numero inteiro")
    elif n <= 0 or n > x.size:
        raise ValueError("n deve estar entre 1 e o tamanho de x")

    soma = np.cumsum(x, dtype=float)
    soma = np.concatenate((np.array([0.0]), soma))
    return (soma[n:] - soma[:-n]) / n


if __name__ == "__main__":
    # x = np.array([2, 4, 6, 8, 10, 12])
    # n = 3
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    n = 4

    print("Vetor:", x)
    print("Tamanho da janela:", n)
    print("Media movel:", media_movel(x, n))
