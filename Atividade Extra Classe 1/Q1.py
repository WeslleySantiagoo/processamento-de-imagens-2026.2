import numpy as np


def contar_transicoes(x):
    """Conta as transicoes de False para True em um vetor booleano."""
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x deve ser um vetor unidimensional")
    if not np.issubdtype(x.dtype, np.bool_):
        raise TypeError("x deve conter apenas valores booleanos")

    transicoes = (~x[:-1]) & x[1:]
    return int(np.sum(transicoes))


if __name__ == "__main__":
    # x = np.array([1, False, True, 0])
    # x = np.array([[1, 2, 3], [4, 5, 6]])
    x = np.array([False, True, False, False, True, True, False, True, True, False])

    print("Sequencia:", x)
    print("Numero de transicoes de False para True:", contar_transicoes(x))
