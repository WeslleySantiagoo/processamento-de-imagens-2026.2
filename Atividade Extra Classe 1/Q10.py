import numpy as np


def esta_ordenado(v):
    """Verifica se um vetor esta ordenado de forma nao decrescente."""
    v = np.asarray(v)

    if v.ndim != 1:
        raise ValueError("v deve ser um vetor unidimensional")
    elif not (np.issubdtype(v.dtype, np.integer) or np.issubdtype(v.dtype, np.floating)):
        raise TypeError("v deve conter apenas numeros reais")

    return bool(np.all(v[:-1] <= v[1:]))


if __name__ == "__main__":
    # v = np.array([1, 3, 2, 4, 5])
    v = np.array([1, 2, 3, 4, 5])

    print("Vetor:", v)
    print("Esta ordenado crescentemente?", esta_ordenado(v))
