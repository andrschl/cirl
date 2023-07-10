# Here we define a few helper functions such as projections, intersections, and complements for linear spaces.

from numpy.linalg import lstsq, matrix_rank, norm
from scipy.linalg import null_space, orth, pinv
import numpy as np

def intersection(A: np.ndarray, B: np.ndarray):
    """
    Compute intersection of column spaces of A and B.
    :param A: size (n,m) array.
    :param B: size (n,l) array.
    :return: intersection.
    """
    return null_space(np.concatenate([null_space(A.T).T, null_space(B.T).T], axis=0))

def orthogonal_complement(A: np.ndarray):
    """
    Get orthogonal complement of column span of A.
    :param A: size (n,m) array.
    :return: orthogonal complement.
    """
    return null_space(A.T)

def unit_vector(v: np.ndarray, axis=None, ord=None):
    """
    Get unit vector.
    :param v: vector or matrix to be normalized.
    :param axis: dimension along to normalize.
    :param ord: choice of norm.
    :return: unit vector.
    """
    return v / norm(v, axis=axis, keepdims=True, ord=ord)

def orthogonal_projection(v, A, is_orth=False):
    """
    Orthogonal projection of v onto column space of A.
    If columns of A are orthonormal we could just do A @A.T @ v.
    :param v: vector to be projected.
    :param A: matrix of columns spanning the subspace.
    :return: orthogonal projection of v onto span(A).
    """
    return A @ A.T @ v if is_orth else A @ pinv(A) @ v