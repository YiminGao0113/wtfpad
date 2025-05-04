import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from dataset import WFDataset
import time

print("Loading datasets...")

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

print("Datasets loaded.")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

print("Converting to numpy arrays...")

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

# Shuffle training data
perm = np.random.permutation(len(X_train))
X_train = X_train[perm]
y_train = y_train[perm]

X_test = np.array(X_test)
y_test = np.array(y_test)

print("Data conversion done.")
print("Training SVM (RBF kernel)...")

start_time = time.time()

clf = SVC(
    kernel='rbf',     # or 'linear' for linear SVM
    C=1.0,
    gamma='scale',
    verbose=True
)

clf.fit(X_train, y_train)
print(f"Training completed in {time.time() - start_time:.2f} seconds.")

print("Evaluating...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {100.0 * acc:.2f}%")
