import numpy as np


def somar_divisiveis(x, divisor):
    """Soma os elementos de um vetor que sao divisiveis pelo valor informado."""
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x deve ser um vetor unidimensional")
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError("x deve conter apenas numeros inteiros")
    if not isinstance(divisor, (int, np.integer)):
        raise TypeError("o divisor deve ser um numero inteiro")
    if divisor == 0:
        raise ValueError("o divisor deve ser diferente de zero")

    divisiveis = x[x % divisor == 0]
    return np.sum(divisiveis)


if __name__ == "__main__":
    x = np.array([1, 5, 12, 15, 20, 22])
    divisor = 5

    print("Vetor:", x)
    print("Divisor escolhido:", divisor)
    print("Soma dos elementos divisiveis:", somar_divisiveis(x, divisor))
