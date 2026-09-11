import numpy as np


def contar_pares(x):
    """Conta quantos valores pares existem em um vetor."""
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x deve ser um vetor unidimensional")
    elif not np.issubdtype(x.dtype, np.integer):
        raise TypeError("x deve conter apenas numeros inteiros")

    return int(np.sum(x % 2 == 0))


if __name__ == "__main__":
    x = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    # x = np.array([1, 2, 3, 4, 5, 6, 7])

    print("Vetor:", x)
    print("Quantidade de valores pares:", contar_pares(x))
