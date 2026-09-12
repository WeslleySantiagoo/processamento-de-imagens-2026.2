import numpy as np


def eh_palindromo(x):
    """Verifica se um vetor possui a mesma ordem quando invertido."""
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x deve ser um vetor unidimensional")

    return bool(np.all(x == x[::-1]))


if __name__ == "__main__":
    # x = np.array([1, 2, 3, 2, 1])
    x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

    print("Vetor:", x)
    print("E um palindromo?", eh_palindromo(x))
