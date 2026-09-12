import numpy as np


def somar_positivos(v):
    """Soma apenas os valores positivos de um vetor."""
    v = np.asarray(v)

    if v.ndim != 1:
        raise ValueError("v deve ser um vetor unidimensional")
    elif not (np.issubdtype(v.dtype, np.integer) or np.issubdtype(v.dtype, np.floating)):
        raise TypeError("v deve conter apenas numeros reais")

    return np.sum(v[v > 0])


if __name__ == "__main__":
    # v = np.array([-4, -1, 0, 2, 5])
    v = np.array([1, -2, 3, 4, -5, 6, -7, 8, -9])

    print("Vetor:", v)
    print("Soma dos valores positivos:", somar_positivos(v))
