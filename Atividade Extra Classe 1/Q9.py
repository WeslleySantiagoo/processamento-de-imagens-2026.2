import numpy as np


def distancia_euclidiana(a, b):
    """Calcula a distancia euclidiana entre dois vetores."""
    a = np.asarray(a)
    b = np.asarray(b)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a e b devem ser vetores unidimensionais")
    elif a.shape != b.shape:
        raise ValueError("a e b devem ter o mesmo tamanho")
    elif not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
        raise TypeError("a e b devem conter apenas numeros")

    return np.sqrt(np.sum((a - b) ** 2))


if __name__ == "__main__":
    # a = np.array([0, 0])
    # b = np.array([3, 4])
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    print("a:", a)
    print("b:", b)
    print("Distancia euclidiana:", distancia_euclidiana(a, b))
