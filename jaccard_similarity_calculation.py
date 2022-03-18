#!/usr/bin/env python
"""
JaccardDistance_SparseMatrix

Calculate Jaccard distance for sparse matrix

Firstly, the jaccard similarity calculation code is created by [na-o-ys (github.com)](https://github.com/na-o-ys), and referenced from [Compute all pairwise vector similarities within a sparse matrix (Python) (na-o-ys.github.io)](http://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html)

I didn't make extensive changes to the source code, just added support for scipy.sparse.csr.csr_matrix and a little bit of other improvements.

Another way to do it is here: [python - Compute Jaccard distances on sparse matrix - Stack Overflow](https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix)
---------------------------------------------------------------
Calculate Jaccard Distance using scipy CSR or CSC sparse matrix
1. Calcualte Jaccard Similarities firstly 
   - --> jaccard_similarity_dense
2. Create numpy matrix has same shape size with jaccard_similarity_dense, filling with 1 
   - --> full_one_matrix 
3. Use `1 - jaccard similarity = jaccard distance` to calculate distance
"""
import numpy as np
import scipy
from scipy.spatial.distance import squareform

def jaccard_similarities(mat):
    """
    Referenced from
    http://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html
    
    Args:
        mat (CSR matrix): the mat is binary (0 or 1) matrix and the type is scipy.sparse.csr_matrix.

    Returns:
        _type_: _description_
    """
    if type(mat) == scipy.sparse.csr.csr_matrix:
        # gets the number of non-zeros per row of origin matrix
        rows_sum = mat.getnnz(axis=1)

        # Use multiplication to get intersection
        ab = mat * mat.T

        aa = np.repeat(rows_sum, ab.getnnz(axis=1))
    elif type(mat) == scipy.sparse.csc.csc_matrix:
        cols_sum = mat.getnnz(axis=0)

        ab = mat.T * mat

        aa = np.repeat(cols_sum, ab.getnnz(axis=0))

    bb = rows_sum[ab.indices]

    similarities = ab.copy()
    
    similarities.data /= (aa + bb - ab.data)

    return similarities

def jaccard_distance_sparse_matrix(sp_mat):
    """
    Jaccard Distance = 1 - Jaccard Similarity
    
    Matrix sp_mat is binary (0 or 1) matrix (one hot encoding) and the type is scipy.sparse.csr_matrix or scipy.sparse.csc_matrix.

    Args:
        sp_mat (sparse matrix): scipy.sparse.csr_matrix or scipy.sparse.csc_matrix
    """
    if sp_mat.dtype != np.float64:
        # ! change data type to float64, else
        # ! `TypeError: No loop matching the specified signature and casting was found for ufunc true_divide` will be raised
        sp_mat = sp_mat.astype(np.float64)
    
    # Calculate jaccard similarities
    sparse_similarities = jaccard_similarities(sp_mat)
    # Convert to dense matrix
    dense_similarities = sparse_similarities.todense()

    # Create a full matrix with 1
    full_one_matrix = np.full((dense_similarities.shape[0], dense_similarities.shape[1]), 1)
    # Convert to matrix
    full_one_matrix = np.asmatrix(full_one_matrix)
    # Calculate distance
    jaccard_distance = np.subtract(full_one_matrix, dense_similarities)

    # Get condensed distance matrix from jaccard_distance
    # A condensed distance matrix is a flat array containing the upper triangular of the distance matrix
    condensed_distance = squareform(jaccard_distance)

    return condensed_distance
