import numpy as np


def diferencas_consecutivas(v):
    """Calcula a diferenca entre cada elemento e seu antecessor."""
    v = np.asarray(v)

    if v.ndim != 1:
        raise ValueError("v deve ser um vetor unidimensional")
    elif not (np.issubdtype(v.dtype, np.integer) or np.issubdtype(v.dtype, np.floating)):
        raise TypeError("v deve conter apenas numeros reais")

    return v[1:] - v[:-1]


if __name__ == "__main__":
    # v = np.array([2, 5, 9, 14])
    v = np.array([10, 7, 4, 3, 2])

    print("Vetor:", v)
    print("Diferencas consecutivas:", diferencas_consecutivas(v))
