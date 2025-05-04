import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataset import WFDataset
import time

print("Loading datasets...")

train_dataset = WFDataset(
    data_dir="src/knn/batch",
    split='train',
    site_num=100,
    inst_num=17,
    test_num=16
)

test_dataset = WFDataset(
    data_dir="src/knn/batch",
    split='test',
    site_num=100,
    inst_num=17,
    test_num=16
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
print("\n First 5 Training Instances:")
for i in range(5):
    print(f"  Sample {i}: Label = {y_train[i]}, Features = {X_train[i][:10]} ...")  # show first 10 features only


X_test = np.array(X_test)
y_test = np.array(y_test)

print("Data conversion done.")
print("Training Random Forest...")

start_time = time.time()
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    verbose=1
)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {100.0 * train_acc:.2f}%")


print(f"Training completed in {time.time() - start_time:.2f} seconds.")

print("Evaluating...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {100.0 * acc:.2f}%")

print("\n Sample Predictions (one per class):")
for i in range(0, 1100, 100):  # assuming 100 classes × 11 test samples = 1100
    print(f"  Sample {i}: Predicted = {y_pred[i]}, Actual = {y_test[i]}")
