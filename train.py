import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import RadioDataset
from src.ag_cnn import AG_CNN
from src.grad_cam import grad_cam

import tqdm
import os
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tabulate import tabulate


DATA_DIR = "data/train"
BATCH_SIZE = 8
LR = 5e-4
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.2
ST_EPOCH = 10
LR_CHANGE = 0.97



full_dataset = RadioDataset(DATA_DIR, train=True)

total_size = len(full_dataset)
train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

print("Dataset sizes:", train_size, val_size, test_size)

g = torch.Generator().manual_seed(42)
train_ds, val_ds, test_ds = random_split(
    full_dataset, [train_size, val_size, test_size], generator=g
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)



model = AG_CNN(in_channels=3, num_classes=3).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)



train_loss_h = []
val_loss_h = []
train_acc_h = []
val_acc_h = []

best_val_acc = 0.0

os.makedirs("models", exist_ok=True)
os.makedirs("gradcam_outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)


for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    epoch_train_loss = 0.0

    for imgs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_train_loss /= len(train_loader)
    train_acc = correct / total

    train_loss_h.append(epoch_train_loss)
    train_acc_h.append(train_acc)


    model.eval()
    correct, total = 0, 0
    epoch_val_loss = 0.0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)

            loss = criterion(logits, labels)
            epoch_val_loss += loss.item()

            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_val_loss /= len(val_loader)
    val_acc = correct / total

    val_loss_h.append(epoch_val_loss)
    val_acc_h.append(val_acc)


    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        for img in imgs:
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
            print(f">>> LR decayed to {optimizer.param_groups[0]['lr']:.6f}")



    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"models/best_model{epoch}.pth")
        print(f">>> Saved best model at epoch {epoch+1}")



    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {epoch_train_loss:.4f} | "
        f"Val Loss: {epoch_val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )


np.save("logs/train_loss.npy", np.array(train_loss_h))
np.save("logs/val_loss.npy", np.array(val_loss_h))
np.save("logs/train_acc.npy", np.array(train_acc_h))
np.save("logs/val_acc.npy", np.array(val_acc_h))


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


table = [
    ["Metric", "Value"],
    ["Accuracy", f"{acc:.4f}"],
    ["Precision", f"{precision:.4f}"],
    ["Recall", f"{recall:.4f}"],
    ["F1-score", f"{f1:.4f}"],
]

print("\nBest Model Performance on Test Set:")
print(tabulate(table, headers="firstrow", tablefmt="grid"))
