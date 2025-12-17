import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import RadioDataset
from src.ag_cnn import AG_CNN
from src.grad_cam import grad_cam
import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tabulate import tabulate
import os


DATA_DIR = "data/train"
BATCH_SIZE = 8
LR = 5e-4
EPOCHS = 30 
VAL_SPLIT = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.2
ST_EPOCH = 10
LR_CHANGE = 0.97

val_acc_h = []

full_dataset = RadioDataset(DATA_DIR, train=True)
total_size = len(full_dataset)
train_size = int(total_size * 0.70)
val_size = int(total_size * 0.15)
test_size = total_size - train_size - val_size
print("Sizes:", train_size, val_size, test_size)

g = torch.Generator()
g.manual_seed(42)
train_ds, val_ds, test_ds = random_split(
    full_dataset, [train_size, val_size, test_size], generator=g
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = AG_CNN(in_channels=3, num_classes=3).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

os.makedirs("models", exist_ok=True)
os.makedirs("gradcam_outputs", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0

    for imgs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Loss is {loss}")


    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_acc_h.append(val_acc)

    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        for img, lbl in zip(imgs, labels):
            _ = grad_cam(
                model,
                img.unsqueeze(0),
                layer_name="conv3",
                save_all=False,
                output_dir="gradcam_outputs",
            )

    if len(val_acc_h) > ST_EPOCH:
        change = val_acc_h[-1] - val_acc_h[-ST_EPOCH - 1]
        if change < THRESHOLD:
            for g in optimizer.param_groups:
                g["lr"] *= LR_CHANGE
            print(f">>> Learning rate decayed to {optimizer.param_groups[0]['lr']:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f">>> Saved best model at epoch {epoch+1}")

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    )

best_model = AG_CNN(in_channels=3, num_classes=3).to(DEVICE)
best_model.load_state_dict(torch.load("models/best_model.pth"))
best_model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = best_model(imgs)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")

table = [["Accuracy", "Precision", "Recall", "F1-score"],
         [f"{acc:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"]]

print("\nBest Model Performance on Test Set:")
print(tabulate(table, headers="firstrow", tablefmt="grid"))
