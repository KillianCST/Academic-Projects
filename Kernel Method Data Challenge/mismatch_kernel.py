import numpy as np
from scipy import sparse
from itertools import combinations, product

###############################
# K-mer Set and Neighbour Functions
###############################

def create_kmer_set(X, k):
    """
    Return a dictionary mapping each k-mer appearing in the dataset to a unique integer index.
    """
    kmer_set = {}
    for x in X:
        for j in range(len(x) - k + 1):
            kmer = x[j:j+k]
            if kmer not in kmer_set:
                kmer_set[kmer] = len(kmer_set)
    return kmer_set

def get_mismatch_neighbors_on_demand(kmer, m, kmer_set):
    """
    Generate neighbors with up to m mismatches that are in the kmer_set.

    """
    letters = "GTAC"
    k = len(kmer)
    
    # Start with the kmer itself (0 mismatches)
    neighbors = [kmer] if kmer in kmer_set else []
    neighbor_indices = [kmer_set[n] for n in neighbors]
    
    # For 1 to m mismatches
    for d in range(1, m + 1):
        for indices in combinations(range(k), d):
            for replacements in product(letters, repeat=d):
                # Skip if any position keeps its original value
                if any(kmer[idx] == rep for idx, rep in zip(indices, replacements)):
                    continue
                
                # Create the new neighbor
                neighbor = list(kmer)
                for idx, rep in zip(indices, replacements):
                    neighbor[idx] = rep
                neighbor_str = "".join(neighbor)
                
                # Only add if the neighbor is in the kmer_set
                if neighbor_str in kmer_set:
                    neighbor_indices.append(kmer_set[neighbor_str])
    
    return neighbor_indices

def precompute_mismatch_table(kmer_set, m):
    """
    Precomputes a dictionary mapping each kmer to the indices of its neighbors in kmer_set.
    """
    mismatch_table = {}
    for kmer in kmer_set:
        mismatch_table[kmer] = get_mismatch_neighbors_on_demand(kmer, m, kmer_set)
    return mismatch_table

###############################
# Embedding Functions
###############################

def build_feature_matrix(X, k, kmer_set, mismatch_table=None):
    """
    Build a sparse feature matrix directly with all sequences in X.
    Each row corresponds to a kmer feature, each column to a sample.
    """
    rows, cols, data = [], [], []
    
    for sample_idx, x in enumerate(X):
        # Process each kmer in the sequence
        for j in range(len(x) - k + 1):
            kmer = x[j:j+k]
            
            # Skip if this kmer is not in our kmer_set
            if kmer not in kmer_set:
                continue
                
            # If we have a mismatch table, use the precomputed neighbors
            if mismatch_table is not None:
                for idx in mismatch_table[kmer]:
                    rows.append(idx)
                    cols.append(sample_idx)
                    data.append(1)
            else:
                # Just use the kmer itself (no mismatches)
                rows.append(kmer_set[kmer])
                cols.append(sample_idx)
                data.append(1)
    
    # Create sparse matrix
    num_features = len(kmer_set)
    num_samples = len(X)
    
    return sparse.csr_matrix((data, (rows, cols)), shape=(num_features, num_samples))

###############################
# Mismatch Kernel Function
###############################

def mismatch_kernel(X1, X2=None, k=3, m=0, normalize=True, precomputed=None):
    """
    Compute the Gram matrix for the mismatch kernel between two sets of sequences.
    
    """
    if X2 is None:
        X2 = X1
        symmetric = True
    else:
        symmetric = False
    
    # Use precomputed values or compute new ones
    if precomputed is not None:
        kmer_set, mismatch_table = precomputed
    else:
        # Combine sequences for a complete kmer set
        all_sequences = list(X1) + ([] if symmetric else list(X2))
        kmer_set = create_kmer_set(all_sequences, k)
        
        # Precompute mismatch table if needed
        mismatch_table = precompute_mismatch_table(kmer_set, m) if m > 0 else None
    
    # Build feature matrices
    X1_features = build_feature_matrix(X1, k, kmer_set, mismatch_table)
    
    if symmetric:
        X2_features = X1_features
    else:
        X2_features = build_feature_matrix(X2, k, kmer_set, mismatch_table)
    
    # Compute the Gram matrix
    G = X1_features.T.dot(X2_features).toarray()
    
    # Normalize if requested
    if normalize:
        X1_norms = np.sqrt(np.array(X1_features.power(2).sum(axis=0)).flatten())
        X2_norms = np.sqrt(np.array(X2_features.power(2).sum(axis=0)).flatten())
        
        # Avoid division by zero
        X1_norms[X1_norms == 0] = 1
        X2_norms[X2_norms == 0] = 1
        
        G = G / X1_norms.reshape(-1, 1) / X2_norms.reshape(1, -1)
    
    return G



