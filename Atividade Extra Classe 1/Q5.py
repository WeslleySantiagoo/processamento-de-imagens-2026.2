import numpy as np


def substituir_negativos_por_zero(x):
    """Substitui os valores negativos de um vetor por zero."""
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x deve ser um vetor unidimensional")
    elif not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)):
        raise TypeError("x deve conter apenas numeros reais")

    return np.where(x < 0, 0, x)


if __name__ == "__main__":
    # x = np.array([-3, -2, -1, 0, 1, 2, 3])
    x = np.array([-12.5, -0.5, 0.0, 4.2, -8.0, 10.0])

    print("Exemplo de vetor:", x)
    print("Entrada:", x)
    print("Saida:", substituir_negativos_por_zero(x))