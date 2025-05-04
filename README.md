# WTF-PAD (Extended) for Network Security Class Project

**Experimental – for research use only. Use with caution.**

This repository contains the source code to simulate [WTF-PAD](https://homes.esat.kuleuven.be/~mjuarezm/index_files/pdf/esorics16.pdf) and reproduce the results from:

```
Toward an Efficient Website Fingerprinting Defense  
M. Juarez, M. Imani, M. Perry, C. Diaz, M. Wright  
ESORICS 2016
```

We extend the original WTF-PAD implementation by:
- Adding new padding configurations: `light_padding` and `heavy_padding`
- Supporting additional machine learning models: Logistic Regression, SVM, and Random Forest
- Including dataset preparation scripts and a reproducibility workflow

---

Original repo is here: https://github.com/wtfpad/wtfpad

## Reproduce Our Evaluation

### 1. Environment and Dataset

- Follow the top-level [`README`](https://github.com/wtfpad/wtfpad) for environment setup.
- Download the closed-world dataset (100 websites × 33 visits each) using instructions in `data/README`.
- Unzip the dataset under `data/`.

### 2. Baseline (Unprotected Traces)

Run the attack and train Logistic Regression:
```bash
./src/knn/run_attack.sh data/closed-world-original/
python3 train_lr.py
```

This extracts features and outputs model accuracy using unprotected traces.

### 3. Protected Dataset (With Padding)

Run the padding simulation (e.g., `heavy_padding`):
```bash
python src/main.py -c heavy_padding data/closed-world-original
```

Then evaluate it:
```bash
./src/knn/run_attack.sh results/heavy_padding_*
python3 train_lr.py
```

Repeat for other configurations (e.g., `normal_rcv`, `light_padding`, etc.).

### 4. Custom Train/Test Split for Random Forest

To adjust the number of samples per class:
- Edit training/testing split variables in `train_rf.py` (default: 33 samples per site).
- Re-run feature extraction and model evaluation using the same steps as above.

---

## Performance Overhead Evaluation

Run the following to evaluate bandwidth and latency overhead:
```bash
python src/overheads.py data/closed-world-original results/heavy_padding_*
```

Example output:
```
Bandwidth overhead: 1.91
Latency overhead: 1.00
```

Interpretation: 91% more bandwidth, no latency increase. Results may vary slightly across runs due to randomness.

---

## Supported ML Models

This repo includes additional scripts to train:
- `train_lr.py` — Logistic Regression
- `train_svm.py` — Support Vector Machine
- `train_rf.py` — Random Forest (with configurable per-class splits)

Each uses features extracted from the batch files produced by `run_attack.sh`.

---

## Advanced Configuration

The padding logic in WTF-PAD uses distributions defined in `config.ini`. For example:

```ini
[heavy_padding]
client_snd_burst_dist = norm, 15, 0.0005, 0.02
client_snd_gap_dist   = norm, 20, 0.005, 0.02
...
```

Each section controls the behavior of client/server padding in both burst and gap states. See the comments in `config.ini` for details.

You can generate new padding traces with any configuration:
```bash
python src/main.py -c <your_config> data/closed-world-original
```

---

## APE

APE is a simplified version of WTF-PAD that avoids histogram tuning and is implemented as a real Tor pluggable transport. It is not included here but is available via:
- [APE Project](https://www.cs.kau.se/pulls/hot/thebasketcase-ape/)
- [basket2 on GitHub](https://github.com/pylls/basket2)

---

## Questions and Contact

Original WTF-PAD authors:
- Marc Juarez (marc.juarez@kuleuven.be)
- Mohsen Imani (imani.moh@gmail.com)

Extended ML evaluation and automation by:
- Yimin Gao — ML model integration, dataset preparation
- Zhenghong Chen — Padding configuration design


---

📌 For further details, see our class report and the figures in the `fig/` directory.
