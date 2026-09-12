import numpy as np


def operacoes(A, B):
    """Calcula operacoes elemento a elemento entre duas matrizes."""
    A = np.asarray(A)
    B = np.asarray(B)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A e B devem ser matrizes")
    elif A.shape != B.shape:
        raise ValueError("A e B devem ter a mesma dimensao")
    elif not (np.issubdtype(A.dtype, np.number) and np.issubdtype(B.dtype, np.number)):
        raise TypeError("A e B devem conter apenas numeros")

    soma = A + B
    diferenca = np.where(A >= B, A - B, B - A)
    media = soma / 2
    maior = np.where(A >= B, A, B)

    return soma, diferenca, media, maior


if __name__ == "__main__":
    n = 10
    # n = 5
    # n = 6
    # n = 2
    A = np.arange(n * n).reshape(n, n)
    B = np.arange(n * n, 0, -1).reshape(n, n)

    soma, diferenca, media, maior = operacoes(A, B)

    print("A:\n", A)
    print("B:\n", B)
    print("Soma:\n", soma)
    print("Diferenca absoluta:\n", diferenca)
    print("Media:\n", media)
    print("Maior valor em cada posicao:\n", maior)
