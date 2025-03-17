import pandas as pd 
import utils as u 
import numpy as np
from kernel_methods import Kernel_SVM
from data_augmentation import augment_dataset
from mismatch_kernel import mismatch_kernel
from kernels import spectrum_kernel
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

# Training/Validation Split 
X_train1, _, y_train1, _ = u.split_data(dataset1, test_size=0., random_state=42, shuffle=False)
X_train2, _, y_train2, _ = u.split_data(dataset2, test_size=0., random_state=42, shuffle=False)
X_train3, _, y_train3, _ = u.split_data(dataset3, test_size=0., random_state=42, shuffle=False)

# Data augmentation
X_train1_aug, y_train1_aug = augment_dataset(X_train1, y_train1)
X_train2_aug, y_train2_aug = augment_dataset(X_train2, y_train2)
X_train3_aug, y_train3_aug = augment_dataset(X_train3, y_train3)


# Classification : Dataset 1 
print("\nClassification : Dataset 1 ....")

# Load precomputed training kernel (4000x4000)
K1_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug1_k11_m2_normalizeTrue.npy"))
K2_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug1_k20_m1_normalizeTrue.npy"))
K3_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug1_k12_m2_normalizeTrue.npy"))
K4_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug1_k5_m1_normalizeTrue.npy"))
K5_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug1_k11_m1_normalizeTrue.npy"))
K6_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug1_k17_m1_normalizeTrue.npy"))
K7_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug1_k10_m1_normalizeTrue.npy"))
K8_train = np.load(os.path.join(base_path, "precomputed_kernels", "spectrum_kernel_aug1_k11.npy"))
K_train = (K1_train + K2_train + K3_train + K4_train + K5_train + K6_train + K7_train + K8_train) / 8

K1_test = mismatch_kernel(X_test1, X_train1_aug, k=11, m=2, normalize=True)
K2_test = mismatch_kernel(X_test1, X_train1_aug, k=20, m=1, normalize=True)
K3_test = mismatch_kernel(X_test1, X_train1_aug, k=12, m=2, normalize=True)
K4_test = mismatch_kernel(X_test1, X_train1_aug, k=5, m=1, normalize=True)
K5_test = mismatch_kernel(X_test1, X_train1_aug, k=11, m=1, normalize=True)
K6_test = mismatch_kernel(X_test1, X_train1_aug, k=17, m=1, normalize=True)
K7_test = mismatch_kernel(X_test1, X_train1_aug, k=10, m=1, normalize=True)
K8_test = spectrum_kernel(X_test1, X_train1_aug, k=11)
K_test = (K1_test + K2_test + K3_test + K4_test + K5_test + K6_test + K7_test + K8_test) / 8

model = Kernel_SVM(kernel='precomputed', C=1.19)
model.fit(K_train, y_train1_aug)
train_score1 = model.score(K_train, y_train1_aug)
print("Dataset 1 Training Accuracy:", train_score1)
pred1 = model.predict(K_test)
print("Dataset 1 Test Predictions: \n", pred1)


# Classification : Dataset 2 
print("\nClassification : Dataset 2 ....")

K1_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug2_k10_m1_normalizeTrue.npy"))
K2_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug2_k17_m1_normalizeTrue.npy"))
K3_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug2_k5_m2_normalizeTrue.npy"))
K4_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug2_k11_m1_normalizeTrue.npy"))
K5_train = np.load(os.path.join(base_path, "precomputed_kernels", "spectrum_kernel_aug2_k13.npy"))
K6_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug2_k9_m1_normalizeTrue.npy"))
K_train = (K1_train + K2_train + K3_train + K4_train + K5_train + K6_train) / 6


K1_test = mismatch_kernel(X_test2, X_train2_aug, k=10, m=1, normalize=True)
K2_test = mismatch_kernel(X_test2, X_train2_aug, k=17, m=1, normalize=True)
K3_test = mismatch_kernel(X_test2, X_train2_aug, k=5, m=2, normalize=True)
K4_test = mismatch_kernel(X_test2, X_train2_aug, k=11, m=1, normalize=True)
K5_test = spectrum_kernel(X_test2, X_train2_aug, k=13)
K6_test = mismatch_kernel(X_test2, X_train2_aug, k=9, m=1, normalize=True)
K_test = (K1_test + K2_test + K3_test + K4_test + K5_test + K6_test) / 6
model = Kernel_SVM(kernel='precomputed', C=1.07)

model.fit(K_train, y_train2_aug)
train_score2 = model.score(K_train, y_train2_aug)
print("Dataset 2 Training Accuracy:", train_score2)
pred2 = model.predict(K_test)
print("Dataset 2 Test Predictions: \n", pred2)


# Classification : Dataset 3 
print("\nClassification : Dataset 3 ....")

K1_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug3_k17_m1_normalizeTrue.npy"))
K2_train = np.load(os.path.join(base_path, "precomputed_kernels", "spectrum_kernel_aug3_k12.npy"))
K3_train = np.load(os.path.join(base_path, "precomputed_kernels", "spectrum_kernel_aug3_k13.npy"))
K4_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug3_k14_m2_normalizeTrue.npy"))
K5_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug3_k19_m1_normalizeTrue.npy"))
K6_train = np.load(os.path.join(base_path, "precomputed_kernels", "mismatch_kernel_aug3_k15_m2_normalizeTrue.npy"))
K_train = (K1_train + K2_train + K3_train + K4_train+ K5_train+ K6_train) / 6


K1_test = mismatch_kernel(X_test3, X_train3_aug, k=17, m=1, normalize=True)
K2_test = spectrum_kernel(X_test3, X_train3_aug, k=12)
K3_test = spectrum_kernel(X_test3, X_train3_aug, k=13)
K4_test = mismatch_kernel(X_test3, X_train3_aug, k=14, m=2, normalize=True)
K5_test = mismatch_kernel(X_test3, X_train3_aug, k=19, m=1, normalize=True)
K6_test = mismatch_kernel(X_test3, X_train3_aug, k=15, m=2, normalize=True)
K_test = (K1_test + K2_test + K3_test + K4_test + K5_test + K6_test) / 6

model = Kernel_SVM(kernel='precomputed', C=0.98)
model.fit(K_train, y_train3_aug)
train_score3 = model.score(K_train, y_train3_aug)
print("Dataset 3 Training Accuracy:", train_score3)
pred3 = model.predict(K_test)
print("Dataset 3 Test Predictions: \n", pred3)

# Submission : 
u.save_submission(pred1, pred2, pred3, filename=os.path.join(base_path, "submission", "submission.csv"))
print("Submission file saved!")
