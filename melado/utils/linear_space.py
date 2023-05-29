import math
import numpy as np
from numpy._typing import NDArray


class RealSpace:
    """Real Space methods"""

    @classmethod
    def inner_product(cls, v: NDArray, u: NDArray) -> float:
        """Standard real dot product `v` * `u`"""
        if v.size != u.size:
            raise Exception("Vectors have different length")
        return sum(x * y for x, y in zip(v, u))

    @classmethod
    def norm(cls, v: NDArray) -> float:
        """Norm of given vector `v`"""
        return math.sqrt(cls.inner_product(v, v))

    @classmethod
    def distance(cls, v: NDArray, u: NDArray) -> float:
        """Distance between vectors `v` and `u`"""
        return math.sqrt(cls.inner_product(v, u))

    @classmethod
    def proj(cls, v: NDArray, u: NDArray) -> NDArray:
        """Projection of `v` in `u`"""
        return cls.inner_product(u, v) / cls.inner_product(u, u) * u

    @classmethod
    def gram_schmidt(cls, set: list[NDArray]) -> list[NDArray]:
        """Gram-Schmidt algorithm.
        Takes a list of vectors (`set`) and applies the method of Gram-Schmidt,
        orthonormalizing `set`.
        """
        # Orthogonalization algorithm
        ortho: list[NDArray] = [np.copy(set[0]) / cls.norm(set[0])]
        for j in range(1, len(set)):
            q = np.copy(set[j])
            for i in range(j):
                r = cls.inner_product(set[j], ortho[i])
                q -= r * ortho[i]

            # Check if the list is LI and, if so, append the normalized vector
            norm = cls.norm(q)
            if norm == 0:
                raise Exception("The list of vectors is linearly dependent.")
            ortho.append(q / norm)

        return ortho
