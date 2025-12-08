import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import RadioDataset
from src.ag_cnn import AG_CNN

DATA_DIR = "data/train"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset Split (առանց train validation) ---
full_dataset = RadioDataset(DATA_DIR, train=True)
total_size = len(full_dataset)
train_size = int(total_size * 0.70)
val_size = int(total_size * 0.15)
test_size = total_size - train_size - val_size

g = torch.Generator()
g.manual_seed(42)
_, _, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=g)

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


model = AG_CNN(in_channels=3, num_classes=3).to(DEVICE)
model.load_state_dict(torch.load("models/best_model.pth"))

model.eval()


total, correct = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Final TEST Accuracy = {test_acc:.4f}")
