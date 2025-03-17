import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from joblib import Parallel, delayed
from scipy.stats import mode 

def accuracy_score(y_true, y_pred):
    """
    Computes the accuracy score: the fraction of samples that were correctly predicted.
    
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    correct = np.sum(y_true == y_pred)
    return correct / y_true.size


def cross_val_score_with_augmentation(estimator, X, y, augment_dataset, cv=5, n_jobs=-1, verbose=0):
    """
    Custom cross-validation function that augments training data only,
    and uses parallel training over folds.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
   
    def fit_and_score(train_idx, test_idx):
        try:
            X_train_cv, y_train_cv = X[train_idx], y[train_idx]
            X_test_cv, y_test_cv = X[test_idx], y[test_idx]
            # Augment the training fold.
            X_train_aug, y_train_aug = augment_dataset(X_train_cv, y_train_cv)
            # Clone the estimator 
            est = clone(estimator)
            est.fit(X_train_aug, y_train_aug)
            return est.score(X_test_cv, y_test_cv)
        except Exception as e:
            print(f"Error in fold: {e}")
            # Return NaN to indicate failure in this fold
            return np.nan
   
    scores = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")(
        delayed(fit_and_score)(train_idx, test_idx) for train_idx, test_idx in skf.split(X, y)
    )
   
    return np.array(scores)


def cross_val_score_with_precomputed_kernel(estimator, K, y, cv=5, random_state=42, n_jobs=-1, verbose=0, n_iter=1):
    """
    Custom cross-validation function that uses a precomputed kernel matrix.
    
    This function repeats the CV process n_iter times and returns the average score.
    Both the outer (n_iter) and inner (CV folds) loops are parallelized.
    """
    N = len(y)  
    M = K.shape[0] // N  # check for augmentation
    
    tasks = []
    for i in range(n_iter):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state + i)
        for train_idx, test_idx in skf.split(np.arange(N), y):
            tasks.append((train_idx, test_idx))
    
    def fit_and_score(train_idx, test_idx):
        try:
            
            train_aug_idx = np.concatenate([train_idx + i * N for i in range(M)])
         
            y_train = np.concatenate([y[train_idx] for _ in range(M)])
            # For testing, use only the raw samples.
            y_test = y[test_idx]
            
            # Extract the corresponding submatrices from the kernel.
            K_train = K[np.ix_(train_aug_idx, train_aug_idx)]
            K_test = K[np.ix_(test_idx, train_aug_idx)]
            
            est = clone(estimator)
            est.fit(K_train, y_train)
            return est.score(K_test, y_test)
        except Exception as e:
            print(f"Error in fold: {e}")
            return np.nan

    all_scores = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")(
        delayed(fit_and_score)(train_idx, test_idx) for train_idx, test_idx in tasks
    )
    
    return np.mean(all_scores)


def cross_val_predict_with_precomputed_kernel(estimator, K, y, cv=5, random_state=42, n_jobs=-1, verbose=0, n_iter=1):
    """
    Custom cross-validation function that uses a precomputed kernel matrix and returns cross-validated predictions.

    """
    N = len(y)  
    M = K.shape[0] // N  
    
    tasks = []
    for i in range(n_iter):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state + i)
        for train_idx, test_idx in skf.split(np.arange(N), y):
            tasks.append((train_idx, test_idx))
    
    def fit_and_predict(train_idx, test_idx):
        try:
            train_aug_idx = np.concatenate([train_idx + i * N for i in range(M)])
            y_train = np.concatenate([y[train_idx] for _ in range(M)])
            # For testing, use only the raw samples.
  
            K_train = K[np.ix_(train_aug_idx, train_aug_idx)]
            K_test = K[np.ix_(test_idx, train_aug_idx)]
            
            est = clone(estimator)
            est.fit(K_train, y_train)
            y_pred = est.predict(K_test)
            return (test_idx, y_pred)
        except Exception as e:
            print(f"Error in fold: {e}")
            return (test_idx, np.array([None] * len(test_idx)))
    
    results = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")(
        delayed(fit_and_predict)(train_idx, test_idx) for train_idx, test_idx in tasks
    )
    
    predictions_dict = {i: [] for i in range(N)}
    for test_idx, y_pred in results:
        for idx, pred in zip(test_idx, y_pred):
            predictions_dict[idx].append(pred)
    
    final_predictions = np.empty(N, dtype=object)
    for i in range(N):
        preds = predictions_dict[i]
        if len(preds) == 0:
            raise ValueError(f"Sample {i} was never predicted in cross-validation!")
        # Filter out any None values.
        preds = [p for p in preds if p is not None]
        if len(preds) == 0:
            raise ValueError(f"Sample {i} only has None predictions in cross-validation!")
        mode_result = mode(preds, axis=None, keepdims=False)
        final_predictions[i] = mode_result.mode  
    
    try:
        final_predictions = final_predictions.astype(y.dtype)
    except Exception as e:
        print("Warning: Could not cast final predictions to the same type as y. Returning as object type.")
    
    return final_predictions
