import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils import ProteinDataset, collate_fn
from model import ProteinClassifier

# ======== PARAMETERS ========
EPOCHS = 20
BATCH_SIZE = 2
LR = 0.001
MAX_LEN = 200
# ============================

# Load dataset
dataset = ProteinDataset("dataset.csv", max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Create model
model = ProteinClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("\n----- Training Started -----\n")

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0.0
    
    for sequences, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

print("\n----- Training Completed -----")
torch.save(model.state_dict(), "protein_cnn.pth")
print("Model saved as protein_cnn.pth\n")
