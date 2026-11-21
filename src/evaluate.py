"""
Evaluation & Robustness Stress-Testing:
Evaluates model on clean test set, generates confusion matrix and ROC-AUC,
and performs the JPEG compression sweep (stress-test).
"""
import os
import io
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import load_trained_model

class JPEGPerturbedDataset(Dataset):
    """Applies on-the-fly JPEG compression to test images."""
    def __init__(self, root_dir, quality=100, transform=None):
        self.samples = []
        self.transform = transform
        self.quality = quality

        for label_idx, class_name in enumerate(['real', 'manipulated']):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        
        # Apply JPEG compression perturbation
        if self.quality < 100:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.quality)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

def run_evaluation(data_dir="./data/processed/test", model_path="./models/best_model.pth", output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating with model: {model_path} on {device}")

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    model = load_trained_model(model_path, device=device)

    # 1. Clean Test Evaluation
    clean_dataset = JPEGPerturbedDataset(data_dir, quality=100, transform=eval_transform)
    clean_loader = DataLoader(clean_dataset, batch_size=32, shuffle=False)

    y_true = []
    y_probs = []

    with torch.no_grad():
        for images, labels in clean_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            y_probs.extend(probs)
            y_true.extend(labels.numpy().flatten())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)

    print("\n" + "="*45)
    print("CLEAN TEST EVALUATION METRICS:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  ROC-AUC:   {auc:.4f}")
    print("="*45)

    # Save predictions
    df_preds = pd.DataFrame({'y_true': y_true, 'y_prob': y_probs, 'y_pred': y_pred})
    df_preds.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    # 2. JPEG Robustness Sweep
    print("\nRunning JPEG Compression Stress-Testing...")
    q_factors = [100, 90, 75, 50, 30]
    sweep_results = []

    for q in q_factors:
        ds = JPEGPerturbedDataset(data_dir, quality=q, transform=eval_transform)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        q_true, q_probs = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                q_probs.extend(probs)
                q_true.extend(labels.numpy().flatten())
        
        q_pred = (np.array(q_probs) >= 0.5).astype(int)
        q_acc = accuracy_score(q_true, q_pred)
        q_rec = recall_score(q_true, q_pred, zero_division=0)
        sweep_results.append({'quality': q, 'accuracy': q_acc, 'recall': q_rec})
        print(f"  -> QF={q:3d}: Accuracy = {q_acc*100:.2f}%, Recall = {q_rec*100:.2f}%")

    df_sweep = pd.DataFrame(sweep_results)
    df_sweep.to_csv(os.path.join(output_dir, "jpeg_robustness_sweep.csv"), index=False)

    # Plot degradation curve
    plt.figure(figsize=(7, 4))
    plt.plot(df_sweep['quality'], df_sweep['accuracy']*100, marker='o', label='Accuracy (%)', linewidth=2)
    plt.plot(df_sweep['quality'], df_sweep['recall']*100, marker='s', label='Fraud Recall (%)', linewidth=2, linestyle='--')
    plt.axvline(x=70, color='gray', linestyle=':', label='Mobile SDK Cutoff (QF=70)')
    plt.xlabel("JPEG Quality Factor")
    plt.ylabel("Performance (%)")
    plt.title("Model Degradation Under Transmission Compression")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "jpeg_degradation_curve.png"), dpi=200)
    print(f"\n✅ Degradation curve saved to {output_dir}/jpeg_degradation_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="./data/processed/test")
    parser.add_argument('--model_path', default="./models/best_model.pth")
    parser.add_argument('--output_dir', default="./results")
    args = parser.parse_args()

    run_evaluation(args.data_dir, args.model_path, args.output_dir)
