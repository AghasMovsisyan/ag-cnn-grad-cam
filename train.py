import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import RadioDataset
from src.ag_cnn import AG_CNN
from src.grad_cam import grad_cam

DATA_DIR = "data/train"
BATCH_SIZE = 8
LR = 5e-4
EPOCHS = 15
VAL_SPLIT = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLD = 0.2
ST_EPOCH = 10
LR_CHANGE = 0.95

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


model = AG_CNN(in_channels=3, num_classes=3).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0

    for imgs, labels in train_loader:
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
        for idx, (img, lbl) in enumerate(zip(imgs, labels)):
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

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    )


    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/best_model.pth")
        print(">>> Saved best model")
