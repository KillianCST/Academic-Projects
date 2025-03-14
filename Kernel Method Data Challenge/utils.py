import numpy as np
import pandas as pd
from cross_validation import cross_val_predict_with_precomputed_kernel
from kernel_methods import Kernel_SVM
from tqdm import tqdm

def load_and_merge(features_file, labels_file):

    if features_file.endswith("mat100.csv"):
        X = pd.read_csv(features_file, sep=r"\s+", header=None)
    else:
        X = pd.read_csv(features_file)
    
    Y = pd.read_csv(labels_file)
    
    # Append the 'Id' and 'Bound' columns from Y to X
    X['Id'] = Y['Id']
    X['Bound'] = Y['Bound']
    
    new_order = ['Id', 'Bound'] + [col for col in X.columns if col not in ['Id', 'Bound']]
    X = X[new_order]
    
    return X

def split_data(df, test_size=0.2, random_state=42, shuffle=True):
    # Determine the indices to use based on the shuffle flag.
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(df))
    else:
        indices = np.arange(len(df))
    
    test_set_size = int(len(df) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    # Extract features and labels
    X = df.drop(['Id', 'Bound'], axis=1)
    y = df['Bound']
    
    # Split into training and test sets
    X_train = X.iloc[train_indices].values
    y_train = y.iloc[train_indices].values
    X_test  = X.iloc[test_indices].values
    y_test  = y.iloc[test_indices].values

    # Squeeze arrays to (n,) if they have a shape of (n, 1).
    if X_train.ndim == 2 and X_train.shape[1] == 1:
        X_train = X_train.ravel()
    if X_test.ndim == 2 and X_test.shape[1] == 1:
        X_test = X_test.ravel()
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
    if y_test.ndim == 2 and y_test.shape[1] == 1:
        y_test = y_test.ravel()
        
    return X_train, X_test, y_train, y_test


def save_submission(pred1, pred2, pred3, filename='submission.csv'):
   
    p1 = np.array(pred1)
    p2 = np.array(pred2)
    p3 = np.array(pred3)
    
    # Concatenate the predictions
    preds = np.concatenate((p1, p2, p3))
    
    df = pd.DataFrame({
        'Id': range(len(preds)),
        'Bound': preds
    })
    
    # Save the DataFrame to CSV 
    df.to_csv(filename, index=False)


def compute_and_save_kernel(kernel_func, name, X_aug, **kwargs):

    kernel_matrix = kernel_func(X_aug, **kwargs)
    
    params_str = "_".join(f"{key}{value}" for key, value in kwargs.items())
    
    file_name = f"precomputed_kernels/{kernel_func.__name__}_{name}_{params_str}.npy"
    
    # Save the kernel matrix to a .npy file.
    np.save(file_name, kernel_matrix)
    return f"Saved {kernel_func.__name__} kernel for {name} with parameters {kwargs} to {file_name}"


def load_kernel(kernel_type, dataset, k=5, m=None, d=5):

    if kernel_type == 'mismatch':
        kernel_file = f"precomputed_kernels/mismatch_kernel_aug{dataset}_k{k}_m{m}_normalizeTrue.npy"
    elif kernel_type == 'spectrum':
        kernel_file = f"precomputed_kernels/spectrum_kernel_aug{dataset}_k{k}.npy"
    elif kernel_type == 'weighted_degree':
        kernel_file = f"precomputed_kernels/weighted_degree_kernel_aug{dataset}_d{d}_normalizeTrue.npy"
    else:
        raise ValueError("Invalid kernel type. Choose 'mismatch' or 'spectrum'.")
    return np.load(kernel_file)


def precompute_cv_predictions(candidates, y):
    candidates_with_preds = []
    for candidate in tqdm(candidates, desc="Precomputing CV predictions"):
        K, info = candidate
        y_pred = cross_val_predict_with_precomputed_kernel(
            estimator=Kernel_SVM(kernel='precomputed', C=1),
            K=K,
            y=y,
            n_jobs=12
        )
        candidates_with_preds.append((K, info, y_pred))
    return candidates_with_preds


