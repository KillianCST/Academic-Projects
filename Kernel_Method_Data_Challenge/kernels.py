import numpy as np
import numpy as np
from scipy import sparse
from collections import defaultdict

def polynomial_kernel(X1, X2, degree=3, gamma=1, coef0=1):
    return (gamma * np.dot(X1, X2.T) + coef0) ** degree

def sigmoid_kernel(X1, X2, gamma=0.1, coef0=0):
    return np.tanh(gamma * np.dot(X1, X2.T) + coef0)

def rbf_kernel(X1, X2, gamma=0.5):
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)


def spectrum_kernel(X1, X2=None, k=3):
    
    if X2 is None:
        X2 = X1
        symmetric = True
    else:
        symmetric = False
    
    # Build a dictionary of all unique k-mers across all sequences
    kmer_to_idx = {}
    
    # Process all sequences to build the k-mer dictionary
    for sequences in ([X1] if symmetric else [X1, X2]):
        for x in sequences:
            if len(x) >= k:
                for i in range(len(x) - k + 1):
                    kmer = x[i:i+k]
                    if kmer not in kmer_to_idx:
                        kmer_to_idx[kmer] = len(kmer_to_idx)
    
    # Build sparse matrices for X1 and X2
    n1, n2 = len(X1), len(X2)
    num_features = len(kmer_to_idx)
    
    # Create sparse matrix for X1
    rows1, cols1, data1 = [], [], []
    for i, x in enumerate(X1):
        if len(x) < k:
            continue
            
        # Count k-mers 
        kmer_counts = defaultdict(int)
        for j in range(len(x) - k + 1):
            kmer = x[j:j+k]
            kmer_counts[kmer] += 1
        
        # Add to sparse matrix components
        for kmer, count in kmer_counts.items():
            if kmer in kmer_to_idx: 
                rows1.append(kmer_to_idx[kmer])
                cols1.append(i)
                data1.append(count)
    
    # Create sparse matrix for X1
    X1_sparse = sparse.csr_matrix((data1, (rows1, cols1)), shape=(num_features, n1))
    
    # If symmetric, reuse X1_sparse for X2_sparse
    if symmetric:
        X2_sparse = X1_sparse
    else:
        # Create sparse matrix for X2
        rows2, cols2, data2 = [], [], []
        for i, x in enumerate(X2):
            if len(x) < k:
                continue
                
            kmer_counts = defaultdict(int)
            for j in range(len(x) - k + 1):
                kmer = x[j:j+k]
                kmer_counts[kmer] += 1
            
            for kmer, count in kmer_counts.items():
                if kmer in kmer_to_idx:
                    rows2.append(kmer_to_idx[kmer])
                    cols2.append(i)
                    data2.append(count)
        
        X2_sparse = sparse.csr_matrix((data2, (rows2, cols2)), shape=(num_features, n2))
    
    # Compute the Gram matrix efficiently using matrix multiplication
    G = (X1_sparse.T @ X2_sparse).toarray()
    
    # Compute self-similarities for normalization (sum of squared counts)
    X1_squared_sum = np.array(X1_sparse.power(2).sum(axis=0)).flatten()
    
    if symmetric:
        X2_squared_sum = X1_squared_sum
    else:
        X2_squared_sum = np.array(X2_sparse.power(2).sum(axis=0)).flatten()
    
    # Compute normalization matrix
    norm_matrix = np.sqrt(np.outer(X1_squared_sum, X2_squared_sum))
    
    norm_matrix[norm_matrix == 0] = 1.0
    
    # Return normalized Gram matrix
    return G / norm_matrix


def weighted_degree_kernel_one(x, y, d, beta=None):
    L = len(x)
    if beta is None:
        beta = [1] * d  # Uniform weights if none provided
    score = 0.0
    for m in range(1, d + 1):
        weight = beta[m - 1]
        for i in range(L - m + 1):
            if x[i:i + m] == y[i:i + m]:
                score += weight
    return score

def weighted_degree_kernel(X, Y=None, d=3, beta=None, normalize=True):

    nX = len(X)
    if Y is None:
        Y = X
    nY = len(Y)
    K = np.zeros((nX, nY))
    
    # Compute the kernel matrix
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            K[i, j] = weighted_degree_kernel_one(x, y, d, beta)
    
    if normalize:
        # Compute self-kernels for X and Y
        diag_X = np.array([weighted_degree_kernel_one(x, x, d, beta) for x in X])
        diag_Y = np.array([weighted_degree_kernel_one(y, y, d, beta) for y in Y])
        epsilon = 1e-8  # Avoid division by zero
        diag_X[diag_X < epsilon] = epsilon
        diag_Y[diag_Y < epsilon] = epsilon
        # Normalize each entry K[i,j]
        for i in range(nX):
            for j in range(nY):
                K[i, j] = K[i, j] / np.sqrt(diag_X[i] * diag_Y[j])
    return K











