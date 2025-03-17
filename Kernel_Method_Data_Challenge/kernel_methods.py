import numpy as np
from functools import partial
from cvxopt import matrix, solvers
from sklearn.base import BaseEstimator, ClassifierMixin
solvers.options['show_progress'] = False
from kernels import polynomial_kernel, sigmoid_kernel, rbf_kernel, spectrum_kernel
from mismatch_kernel import mismatch_kernel

###############################################
# Kernel Ridge
###############################################

class Kernel_Ridge(BaseEstimator, ClassifierMixin):
    def __init__(self, lambd=1, kernel='rbf', sigma=1, degree=3, gamma=1, coef0=1,
                 k=3, m=0, normalize=False):
        self.lambd = lambd
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.k = k 
        self.m = m 
        self.normalize = normalize  
        self._set_kernel_fn()  

    def _set_kernel_fn(self):
        if self.kernel == 'rbf':
            self.kernel_fn = partial(rbf_kernel, gamma=self.gamma)
        elif self.kernel == 'polynomial':
            self.kernel_fn = partial(polynomial_kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'sigmoid':
            self.kernel_fn = partial(sigmoid_kernel, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'spectrum':
            self.kernel_fn = partial(spectrum_kernel, k=self.k)
        elif self.kernel == 'mismatch':
            self.kernel_fn = partial(mismatch_kernel, k=self.k, m=self.m, normalize=self.normalize)
        else:
            raise ValueError("Unknown kernel type")
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self._set_kernel_fn()
        return self

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        n_samples = self.X_train.shape[0]
        K = self.kernel_fn(self.X_train, self.X_train)  
        A = K + self.lambd * np.eye(n_samples)
        try:
            self.alpha = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            self.alpha = np.linalg.pinv(A) @ y
        return self

    def predict(self, X):
        K_test = self.kernel_fn(np.asarray(X), self.X_train)
        return (np.dot(K_test, self.alpha) > 0.5).astype(int)

###############################################
# Kernel SVM
###############################################

class Kernel_SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, kernel='rbf', sigma=1, degree=3, gamma=1, coef0=1,
                 k=3, m=0, normalize=False):
        self.C = C
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.k = k 
        self.m = m  
        self.normalize = normalize
        self._set_kernel_fn()

    def _set_kernel_fn(self):
        if self.kernel == 'precomputed':
            self.kernel_fn = None
        elif self.kernel == 'rbf':
            self.kernel_fn = partial(rbf_kernel, gamma=self.gamma)
        elif self.kernel == 'polynomial':
            self.kernel_fn = partial(polynomial_kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'sigmoid':
            self.kernel_fn = partial(sigmoid_kernel, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'spectrum':
            self.kernel_fn = partial(spectrum_kernel, k=self.k)
        elif self.kernel == 'mismatch':
            self.kernel_fn = partial(mismatch_kernel, k=self.k, m=self.m, normalize=self.normalize)
        else:
            raise ValueError("Unknown kernel type")
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self._set_kernel_fn()
        return self

    def fit(self, X, y):
        if self.kernel == 'precomputed':
            # X is assumed to be a precomputed kernel matrix.
            self.K_train = np.asarray(X)
            # For precomputed kernels, we don't need the original X_train.
            self.X_train = None
        else:
            self.X_train = np.asarray(X)
            self.K_train = self.kernel_fn(self.X_train, self.X_train)
        
        # Convert labels: map {0,1} to {-1,1}
        self.y_train = 2 * np.asarray(y, dtype=np.float64).flatten() - 1
        n_samples = self.K_train.shape[0]
        y_col = self.y_train.reshape(-1, 1)
        P_np = self.K_train * (y_col @ y_col.T)
        q_np = -np.ones(n_samples)
        G_np = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h_np = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))
        A_np = self.y_train.reshape(1, -1)
        b_np = np.array([0.])
        P_cv = matrix(P_np)
        q_cv = matrix(q_np)
        G_cv = matrix(G_np)
        h_cv = matrix(h_np)
        A_cv = matrix(A_np, tc='d')
        b_cv = matrix(b_np, tc='d')
        sol = solvers.qp(P_cv, q_cv, G_cv, h_cv, A_cv, b_cv)
        alphas = np.array(sol['x']).flatten()
        self.alpha = alphas
        tol = 1e-5
        sv = (alphas > tol) & (alphas < self.C - tol)
        if np.any(sv):
            b_all = self.y_train[sv] - (self.K_train[:, sv].T @ (alphas * self.y_train))
            self.b = np.mean(b_all)
        else:
            self.b = 0.0
        return self

    def decision_function(self, X):
        if self.kernel == 'precomputed':
            K_test = np.asarray(X)
        else:
            K_test = self.kernel_fn(np.asarray(X), self.X_train)
        # Compute the decision function: dot product of kernel and (alpha * y_train) plus bias.
        decision = np.dot(K_test, self.alpha * self.y_train) + self.b
        return decision

    def predict(self, X):
        decision = self.decision_function(X)
        pred = np.where(decision >= 0, 1, -1)
        return ((pred + 1) // 2).astype(int)

###############################################
# Kernel Logistic Regression
###############################################

class Kernel_Logistic(BaseEstimator, ClassifierMixin):
    def __init__(self, lambd=1, kernel='rbf', sigma=1, degree=3, gamma=1, coef0=1,
                 max_iter=100, tol=1e-5, k=3, m=0, normalize=False):
        self.lambd = lambd
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.k = k  
        self.m = m  
        self.normalize = normalize
        self._set_kernel_fn()
    
    def _set_kernel_fn(self):
        if self.kernel == 'rbf':
            self.kernel_fn = partial(rbf_kernel, gamma=self.gamma)
        elif self.kernel == 'polynomial':
            self.kernel_fn = partial(polynomial_kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'sigmoid':
            self.kernel_fn = partial(sigmoid_kernel, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'spectrum':
            self.kernel_fn = partial(spectrum_kernel, k=self.k)
        elif self.kernel == 'mismatch':
            self.kernel_fn = partial(mismatch_kernel, k=self.k, m=self.m, normalize=self.normalize)
        else:
            raise ValueError("Unknown kernel type")
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self._set_kernel_fn()
        return self

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y, dtype=np.float64).flatten()
        n_samples = self.X_train.shape[0]
        K = self.kernel_fn(self.X_train, self.X_train)
        Z = np.hstack([K, np.ones((n_samples, 1))])
        beta = np.zeros(n_samples + 1)
        for it in range(self.max_iter):
            f = np.dot(Z, beta)
            p = 1 / (1 + np.exp(-f))
            W = np.diag(p * (1 - p))
            alpha = beta[:n_samples]
            g = np.dot(Z.T, (p - self.y_train)) + np.hstack([self.lambd * np.dot(K, alpha), 0])
            H = np.dot(Z.T, np.dot(W, Z))
            H[:n_samples, :n_samples] += self.lambd * K
            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H).dot(g)
            beta_new = beta - delta
            if np.linalg.norm(delta) < self.tol:
                beta = beta_new
                break
            beta = beta_new
        self.alpha_ = beta[:n_samples]
        self.b_ = beta[n_samples]
        return self

    def decision_function(self, X):
        X = np.asarray(X)
        K_test = self.kernel_fn(X, self.X_train)
        return np.dot(K_test, self.alpha_) + self.b_

    def predict_proba(self, X):
        f = self.decision_function(X)
        p = 1 / (1 + np.exp(-f))
        return np.vstack([1-p, p]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y).flatten())







