import numpy as np


def somar_blocos(x, n):
    """Soma cada bloco de n valores consecutivos de um vetor."""
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x deve ser um vetor unidimensional")
    elif not isinstance(n, (int, np.integer)):
        raise TypeError("n deve ser um numero inteiro")
    elif n <= 0:
        raise ValueError("n deve ser maior que zero")
    elif x.size % n != 0:
        raise ValueError("o tamanho de x deve ser divisivel por n")

    blocos = x.reshape(-1, n)
    return np.sum(blocos, axis=1)


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    n = 6

    print("Vetor:", x)
    print("Tamanho de cada bloco:", n)
    print("Soma de cada bloco:", somar_blocos(x, n))
