import numpy as np


def normalizar(x):
    """Normaliza um vetor para que a soma dos elementos seja igual a um."""
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x deve ser um vetor unidimensional")
    elif not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)):
        raise TypeError("x deve conter apenas numeros reais")

    soma = np.sum(x)

    if soma == 0:
        raise ValueError("a soma dos elementos de x deve ser diferente de zero")

    return x / soma


if __name__ == "__main__":
    x = np.array([2, 3, 5, 7, 8, 10, 12, 15, 18, 20])
    # x = np.array([1, 2, 3, 4])

    print("Vetor:", x)
    print("Vetor normalizado:", normalizar(x))
