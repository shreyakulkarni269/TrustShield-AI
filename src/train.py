"""
Training Pipeline for EfficientNet-B0:
Fine-tunes pretrained EfficientNet-B0 for binary classification (Real vs Manipulated),
computes validation loss/accuracy, saves best model checkpoint.
"""
import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import build_efficientnet_b0

def get_data_loaders(data_dir="./data/processed", batch_size=32):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value='random')
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, train_set.class_to_idx

def train_model(data_dir="./data/processed", output_dir="./models", epochs=15, batch_size=32, lr=1e-4):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = get_data_loaders(data_dir, batch_size)
    print(f"Classes: {class_to_idx}")

    model = build_efficientnet_b0(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_acc = train_correct / total_train
        avg_train_loss = train_loss / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = val_correct / total_val
        avg_val_loss = val_loss / total_val
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model to {best_model_path}")

    print("\n✅ Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="./data/processed")
    parser.add_argument('--output_dir', default="./models")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_model(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr)
