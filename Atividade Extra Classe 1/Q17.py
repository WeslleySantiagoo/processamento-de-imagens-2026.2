import numpy as np


def normalizar(A):
    """Normaliza todos os valores de uma matriz para o intervalo [0, 1]."""
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("A deve ser uma matriz")
    elif A.size == 0:
        raise ValueError("A nao pode estar vazia")
    elif not (np.issubdtype(A.dtype, np.integer) or np.issubdtype(A.dtype, np.floating)):
        raise TypeError("A deve conter apenas numeros reais")

    minimo = np.min(A)
    maximo = np.max(A)

    if minimo == maximo:
        raise ValueError("A deve ter pelo menos dois valores diferentes")

    return (A - minimo) / (maximo - minimo), minimo, maximo


def normalizar_linhas(A):
    """Normaliza cada linha de uma matriz para o intervalo [0, 1]."""
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError("A deve ser uma matriz")
    elif A.size == 0:
        raise ValueError("A nao pode estar vazia")
    elif not (np.issubdtype(A.dtype, np.integer) or np.issubdtype(A.dtype, np.floating)):
        raise TypeError("A deve conter apenas numeros reais")

    minimo = np.min(A, axis=1, keepdims=True)
    maximo = np.max(A, axis=1, keepdims=True)

    if np.any(minimo == maximo):
        raise ValueError("cada linha de A deve ter pelo menos dois valores diferentes")

    return (A - minimo) / (maximo - minimo)


if __name__ == "__main__":
    linhas = 10
    colunas = 10
    # linhas = 4
    # colunas = 5
    A = np.arange(linhas * colunas, dtype=float).reshape(linhas, colunas)

    normalizada, minimo, maximo = normalizar(A)
    normalizada_linhas = normalizar_linhas(A)

    print("A:\n", A)
    print("Minimo:", minimo)
    print("Maximo:", maximo)
    print("A normalizada:\n", normalizada)
    print("A normalizada por linha:\n", normalizada_linhas)
