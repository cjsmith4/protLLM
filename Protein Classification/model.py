import torch
import torch.nn as nn

class ProteinClassifier(nn.Module):
    def __init__(self, vocab_size=21, embedding_dim=32, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.cnn = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.fc = nn.Linear(64 * 98, num_classes)  # adjust for MAX_LEN=200

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
