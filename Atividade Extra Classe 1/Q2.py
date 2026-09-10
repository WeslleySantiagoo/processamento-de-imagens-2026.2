import numpy as np


def quadrados_dos_impares(x):
    """Seleciona os numeros impares de um vetor e os eleva ao quadrado."""
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x deve ser um vetor unidimensional")
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError("x deve conter apenas numeros inteiros")

    impares = x[x % 2 != 0]
    return impares**2


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    print("Vetor:", x)
    print("Quadrados dos numeros impares:", quadrados_dos_impares(x))
