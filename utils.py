import torch
from torch.utils.data import Dataset

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}

LABEL_MAP = {"enzyme": 0, "structural": 1}

def encode_sequence(seq, max_len=200):
    encoded = [AA_TO_IDX.get(a, 0) for a in seq]
    encoded = encoded[:max_len]
    encoded += [0] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.long)

class ProteinDataset(Dataset):
    def __init__(self, csv_file, max_len=200):
        self.data = []
        with open(csv_file, "r") as f:
            next(f)
            for line in f:
                seq, label = line.strip().split(",")
                self.data.append((seq, label))
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        return encode_sequence(seq, self.max_len), LABEL_MAP[label]

def collate_fn(batch):
    sequences = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return sequences, labels
