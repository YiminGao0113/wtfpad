import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dataset import WFDataset
import time

print("🔄 Loading datasets...")

train_dataset = WFDataset(
    data_dir="src/knn/batch",
    split='train',
    site_num=100,
    inst_num=22,
    test_num=11
)

test_dataset = WFDataset(
    data_dir="src/knn/batch",
    split='test',
    site_num=100,
    inst_num=22,
    test_num=11
)

print("✅ Datasets loaded.")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

print("🔄 Converting to numpy arrays...")

X_train, y_train = [], []
for idx, (x, y) in enumerate(train_dataset):
    x = (x - x.mean()) / (x.std() + 1e-6)
    X_train.append(x.numpy())
    y_train.append(y)
    if idx % 500 == 0:
        print(f"  Processed {idx} train samples...")

X_test, y_test = [], []
for idx, (x, y) in enumerate(test_dataset):
    x = (x - x.mean()) / (x.std() + 1e-6)
    X_test.append(x.numpy())
    y_test.append(y)
    if idx % 500 == 0:
        print(f"  Processed {idx} test samples...")

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

print("✅ Data conversion done.")

# -------------------------
# Quick summary of X_train and y_train
# -------------------------

print("🔎 Shape of X_train:", X_train.shape)     # e.g., (2200, 735)
print("🔎 Shape of y_train:", y_train.shape)     # e.g., (2200,)

# Print class distribution
unique_classes, counts = np.unique(y_train, return_counts=True)
print("📊 Class distribution in y_train:")
for cls, cnt in zip(unique_classes, counts):
    print(f"  Class {cls:3d}: {cnt} samples")

# Print feature-wise stats
print("\n📈 Feature statistics (first 5 features):")
for i in range(5):
    feature_vals = X_train[:, i]
    print(f"  Feature {i}: mean={feature_vals.mean():.3f}, std={feature_vals.std():.3f}, min={feature_vals.min():.3f}, max={feature_vals.max():.3f}")
print("\n🔍 First 5 training examples (truncated to 10 features each):")
for i in range(50):
    x = X_test[i]
    y = y_test[i]
    print(f"Sample {i} - Label: {y} - Features: {np.round(x[:10], 3)} ...")
