import pandas as pd 
import utils as u 
import numpy as np
from kernel_methods import Kernel_SVM
from data_augmentation import augment_dataset
from mismatch_kernel import mismatch_kernel
import os

# Define the base path
base_path = "/Users/1411/Desktop/Kernel_Method_Challenge"

# Data Loading 
dataset1 = u.load_and_merge(os.path.join(base_path, "data", "Xtr0.csv"), os.path.join(base_path, "data", "Ytr0.csv"))
dataset2 = u.load_and_merge(os.path.join(base_path, "data", "Xtr1.csv"), os.path.join(base_path, "data", "Ytr1.csv"))
dataset3 = u.load_and_merge(os.path.join(base_path, "data", "Xtr2.csv"), os.path.join(base_path, "data", "Ytr2.csv"))

# Test data loading
X_test1 = pd.read_csv(os.path.join(base_path, "data", "Xte0.csv")).drop(['Id'], axis=1).squeeze().values
X_test2 = pd.read_csv(os.path.join(base_path, "data", "Xte1.csv")).drop(['Id'], axis=1).squeeze().values
X_test3 = pd.read_csv(os.path.join(base_path, "data", "Xte2.csv")).drop(['Id'], axis=1).squeeze().values

# Training/Validation Split (all data is used for training in this challenge)
X_train1, _, y_train1, _ = u.split_data(dataset1, test_size=0., random_state=42, shuffle=False)
X_train2, _, y_train2, _ = u.split_data(dataset2, test_size=0., random_state=42, shuffle=False)
X_train3, _, y_train3, _ = u.split_data(dataset3, test_size=0., random_state=42, shuffle=False)

# Data augmentation: returns augmented data and corresponding duplicated labels.
X_train1_aug, y_train1_aug = augment_dataset(X_train1, y_train1)
X_train2_aug, y_train2_aug = augment_dataset(X_train2, y_train2)
X_train3_aug, y_train3_aug = augment_dataset(X_train3, y_train3)


# Classification : Dataset 1 
print("\nClassification : Dataset 1 ....")

# Load precomputed training kernel (4000x4000)
K_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug1_k11_m2_normalizeTrue.npy"))
# Compute test kernel between augmented training data and test samples.
K_test = mismatch_kernel(X_test1, X_train1_aug, k=11, m=2, normalize=True)

model = Kernel_SVM(kernel='precomputed', C=1)
model.fit(K_train, y_train1_aug)
train_score1 = model.score(K_train, y_train1_aug)
print("Dataset 1 Training Accuracy:", train_score1)
pred1 = model.predict(K_test)
print("Dataset 1 Test Predictions:", pred1)


# Classification : Dataset 2 
print("\nClassification : Dataset 2 ....")

K_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug2_k10_m1_normalizeTrue.npy"))
K_test = mismatch_kernel(X_test2, X_train2_aug, k=10, m=1, normalize=True)

model = Kernel_SVM(kernel='precomputed', C=1)
model.fit(K_train, y_train2_aug)
train_score2 = model.score(K_train, y_train2_aug)
print("Dataset 2 Training Accuracy:", train_score2)
pred2 = model.predict(K_test)
print("Dataset 2 Test Predictions:", pred2)


# Classification : Dataset 3 
print("\nClassification : Dataset 3 ....")

K_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug3_k17_m1_normalizeTrue.npy"))
K_test = mismatch_kernel(X_test3, X_train3_aug, k=17, m=1, normalize=True)

model = Kernel_SVM(kernel='precomputed', C=1)
model.fit(K_train, y_train3_aug)
train_score3 = model.score(K_train, y_train3_aug)
print("Dataset 3 Training Accuracy:", train_score3)
pred3 = model.predict(K_test)
print("Dataset 3 Test Predictions:", pred3)

# Submission : 
u.save_submission(pred1, pred2, pred3, filename=os.path.join(base_path, "submission", "submission.csv"))
print("Submission file saved!")
