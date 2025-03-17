import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from cross_validation import mode
from joblib import Parallel, delayed
from kernel_methods import Kernel_SVM


###############################################
# Bagging method
###############################################


class Bagging_Kernel(BaseEstimator, ClassifierMixin):
    """
    Standard bagging ensemble for kernel methods with data augmentation.

    """
    def __init__(self, base_estimator=None, n_estimators=10, augment_data=True, 
                 bootstrap=True, random_state=None, n_jobs=-1):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.augment_data = augment_data
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _fit_estimator(self, K, y, indices, raw_count):
        """
        Fit a single base estimator on a bootstrap sample.

        """
        if self.augment_data:
            # For each raw sample index, add its augmentation (offset by raw_count)
            train_idx_final = np.concatenate([indices, indices + raw_count])
            # Use labels from raw and augmented parts.
            y_train_final = np.concatenate([y[indices], y[indices + raw_count]])
        else:
            train_idx_final = indices
            y_train_final = y[indices]
        K_train = K[np.ix_(train_idx_final, train_idx_final)]
        est = clone(self.base_estimator)
        est.fit(K_train, y_train_final)
        return est, indices

    def fit(self, K, y):
        """
        Fit the bagging ensemble using a precomputed kernel matrix and labels.

        """
        y = np.asarray(y)
        n_total = len(y)
        if self.augment_data:
            if n_total % 2 != 0:
                raise ValueError("When augment_data=True, the number of samples must be even.")
            self.raw_count_ = n_total // 2
        else:
            self.raw_count_ = n_total
        self.n_total_ = n_total  # store total number of samples

        # Use raw_count for sampling if augmentation is enabled.
        n_samples = self.raw_count_
        if self.base_estimator is None:
            from kernel_methods import Kernel_SVM  # adjust as needed
            self.base_estimator = Kernel_SVM(kernel='precomputed', C=1)
        rng = np.random.RandomState(self.random_state)
        indices_list = []
        for i in range(self.n_estimators):
            if self.bootstrap:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            indices_list.append(indices)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_estimator)(K, y, indices, self.raw_count_) for indices in indices_list
        )
        self.estimators_ = [res[0] for res in results]
        self.estimators_samples_ = [res[1] for res in results]
        return self

    def predict(self, K_test_global):
        """
        Predict labels for test samples by aggregating predictions from base estimators.

        """
        K_test_global = np.asarray(K_test_global)
        predictions = []
        for est, indices in zip(self.estimators_, self.estimators_samples_):
            if self.augment_data:
                train_idx_final = np.concatenate([indices, indices + self.raw_count_])
            else:
                train_idx_final = indices
            K_test = K_test_global[:, train_idx_final]
            pred = est.predict(K_test)
            predictions.append(pred)
        predictions = np.array(predictions)  # shape: (n_estimators, n_test)
        maj_vote, _ = mode(predictions, axis=0)
        return maj_vote.flatten()

    def score(self, K_test_global, y_true):
        """
        Compute the accuracy of the ensemble on test data.

        """
        y_pred = self.predict(K_test_global)
        return np.mean(y_pred == np.asarray(y_true))

    def oob_score(self, K_global, y):
        """
        Compute the out-of-bag (OOB) accuracy of the ensemble.

        """
        y = np.asarray(y)
        n_samples = self.raw_count_ if self.augment_data else len(y)

        def _oob_for_estimator(est, indices):
            unique = np.unique(indices)
            oob_idx = np.setdiff1d(np.arange(n_samples), unique)
            if self.augment_data:
                train_idx_final = np.concatenate([indices, indices + self.raw_count_])
            else:
                train_idx_final = indices
            if oob_idx.size > 0:
                K_oob = K_global[oob_idx, :][:, train_idx_final]
                preds = est.predict(K_oob)
                return dict(zip(oob_idx, preds))
            else:
                return {}

        oob_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_oob_for_estimator)(est, indices)
            for est, indices in zip(self.estimators_, self.estimators_samples_)
        )
        oob_agg = {i: [] for i in range(n_samples)}
        for res in oob_results:
            for i, pred in res.items():
                oob_agg[i].append(pred)
        all_preds = []
        all_true = []
        for i in range(n_samples):
            if oob_agg[i]:
                maj, _ = mode(np.array(oob_agg[i]))
                maj = np.atleast_1d(maj)
                all_preds.append(maj[0])
                all_true.append(y[i])
        if not all_preds:
            return None
        return np.mean(np.array(all_preds) == np.array(all_true))


class Hierarchical_Bagging_Kernel(Bagging_Kernel):
    """
    Bagging ensemble with an additional meta-layer SVM that aggregates the 
    decision functions of the base estimators.

    """
    def __init__(self, base_estimator=None, n_estimators=10, augment_data=True, 
                 bootstrap=True, random_state=None, n_jobs=-1, meta_estimator=None):
        """
        Parameters are as in Bagging_Kernel with an additional:

        meta_estimator : object, default=None
            The estimator used as the meta classifier. If None, a Kernel_SVM 
            (with kernel='precomputed') is used.
        """
        super().__init__(base_estimator=base_estimator, n_estimators=n_estimators,
                         augment_data=augment_data, bootstrap=bootstrap, 
                         random_state=random_state, n_jobs=n_jobs)
        self.meta_estimator = meta_estimator

    def fit(self, K, y):
        """
        First trains the base estimators using the bagging procedure, then
        creates meta features using the decision functions of the base estimators,
        and finally trains the meta SVM.

        """
        super().fit(K, y)
        raw_count = self.raw_count_
        # Build meta feature matrix: one feature per base estimator for each raw sample.
        meta_features = np.zeros((raw_count, self.n_estimators))
        for j, (est, indices) in enumerate(zip(self.estimators_, self.estimators_samples_)):
            if self.augment_data:
                train_idx_final = np.concatenate([indices, indices + raw_count])
            else:
                train_idx_final = indices
            # Use the precomputed kernel rows corresponding to the raw samples.
            K_train_subset = K[:raw_count, :][:, train_idx_final]
            meta_features[:, j] = est.decision_function(K_train_subset)
        if self.meta_estimator is None:
            from kernel_methods import Kernel_SVM  # adjust as needed
            self.meta_estimator = Kernel_SVM(kernel='precomputed', C=1)
        # Compute a linear kernel on the meta features.
        meta_kernel = np.dot(meta_features, meta_features.T)
        # Use only the raw labels (first raw_count entries) for meta training.
        self.meta_estimator.fit(meta_kernel, y[:raw_count])
        self._meta_features_train = meta_features
        return self

    def predict(self, K_test_global):

        K_test_global = np.asarray(K_test_global)
        n_test = K_test_global.shape[0]
        meta_features_test = np.zeros((n_test, self.n_estimators))
        for j, (est, indices) in enumerate(zip(self.estimators_, self.estimators_samples_)):
            if self.augment_data:
                train_idx_final = np.concatenate([indices, indices + self.raw_count_])
            else:
                train_idx_final = indices
            K_test_subset = K_test_global[:, train_idx_final]
            meta_features_test[:, j] = est.decision_function(K_test_subset)
        meta_kernel_test = np.dot(meta_features_test, self._meta_features_train.T)
        return self.meta_estimator.predict(meta_kernel_test)



###############################################
# Ensemble method
###############################################


class EnsembleKernel(BaseEstimator, ClassifierMixin):
    """
    An ensemble method for kernel-based models that combines multiple base models
    trained on different precomputed kernel matrices.
    
    The final prediction is computed via majority vote across all models.
    """
    def __init__(self, models, n_jobs=-1):
        """
        Parameters
        ----------
        models : list of estimators
            A list of base models (e.g., Kernel_SVM instances) that support fit/predict with precomputed kernels.
        n_jobs : int, default=-1
            Number of jobs to run in parallel during fitting.
        """
        self.models = models
        self.n_jobs = n_jobs

    def fit(self, train_kernels, y_train_aug):
        """
        Fit each base model on each provided training kernel.
        
        Parameters
        ----------
        train_kernels : list of numpy.ndarray, each of shape (M, M)
            List of precomputed training kernel matrices. M is the number of training samples 
            in the augmented set (typically, M = 2*N if augmented labels are duplicated).
        y_train_aug : numpy.ndarray of shape (M,)
            The training labels corresponding to the augmented training data.
            
        """
        tasks = []
        for k_idx, K_train in enumerate(train_kernels):
            for model in self.models:
                tasks.append((k_idx, clone(model), K_train))
        
        def fit_task(k_idx, model, K_train):
            model.fit(K_train, y_train_aug)
            return (model, k_idx)
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_task)(k_idx, model, K_train) for k_idx, model, K_train in tasks
        )
        
        # Store the fitted models and associated training kernel index.
        self.fitted_models_ = results  
        return self

    def predict(self, test_kernels):
        
        predictions = []
        for model, k_idx in self.fitted_models_:
            K_test = test_kernels[k_idx]
            preds = model.predict(K_test)
            predictions.append(preds)
        predictions = np.array(predictions)  # shape: (n_models, n_test)
        majority_preds, _ = mode(predictions, axis=0, keepdims=False)
        return majority_preds.flatten()
