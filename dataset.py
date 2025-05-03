import torch
from torch.utils.data import Dataset
import os

class WFDataset(Dataset):
    def __init__(self, data_dir, split='train', site_num=10, inst_num=5, test_num=5):
        self.samples = []
        self.labels = []

        for site_id in range(site_num):
            if split == 'train':
                count = inst_num
                offset = 0
            elif split == 'test':
                count = test_num
                offset = inst_num
            else:
                raise ValueError("split must be 'train' or 'test'")

            real_inst = 0
            found = 0
            while found < count:
                fname = os.path.join(data_dir, f"{site_id}-{real_inst}f")
                if os.path.exists(fname):
                    with open(fname) as f:
                        line = f.readline().strip()
                    feats = [float(x) if "X" not in x else -1.0 for x in line.split()]
                    self.samples.append(feats)
                    self.labels.append(site_id)
                    found += 1
                real_inst += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
