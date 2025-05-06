import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import numpy as np
import random
import csv
import os

from pet_dataset import PetDataset
from unet import UNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def compute_iou_and_dice(pred, target, num_classes=3):
    valid_mask = (target != 3)
    pred = pred[valid_mask]
    target = target[valid_mask]

    total_inter, total_union, total_dice_num, total_dice_den = [0.0]*3, [0.0]*3, [0.0]*3, [0.0]*3
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        inter = (pred_cls & target_cls).float().sum().item()
        union = (pred_cls | target_cls).float().sum().item()
        dice_den = pred_cls.float().sum().item() + target_cls.float().sum().item()
        total_inter[cls] += inter
        total_union[cls] += union
        total_dice_num[cls] += 2 * inter
        total_dice_den[cls] += dice_den
    return total_inter, total_union, total_dice_num, total_dice_den

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = "./Dataset"
    full_dataset = PetDataset(data_path)

    subset_size = min(200, len(full_dataset)) 
    subset = Subset(full_dataset, range(subset_size))

    total = len(subset)
    train_size = round(0.8 * total)
    val_size = total - train_size

    print(f"Subset size: {total} → Train: {train_size}, Val: {val_size}")

    train_dataset, val_dataset = random_split(subset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    sweep_weights = [
        torch.tensor([1.0, 1.0, 1.0, 1.0]),  
        torch.tensor([1.0, 2.0, 1.0, 3.0]),  
        torch.tensor([1.0, 3.0, 2.0, 0.5]),  
        torch.tensor([1.0, 2.0, 0.7, 3.0]),  
        torch.tensor([1.0, 2.0, 1,0, 2.0]),  
    ]

    for i, class_weights in enumerate(sweep_weights):
        print(f"\nSweep {i+1}: class_weights = {class_weights.tolist()}")

        model = UNet(in_channels=3, num_classes=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))


        best_val_loss = float('inf')
        epochs = 15

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for img, mask in train_loader:
                img = img.to(device)
                mask = mask.squeeze(1).long().to(device)

                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, mask)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                total_inter, total_union, total_dice_num, total_dice_den = [0.0]*3, [0.0]*3, [0.0]*3, [0.0]*3
                for img, mask in val_loader:
                    img = img.to(device)
                    mask = mask.squeeze(1).long().to(device)
                    output = model(img)
                    loss = criterion(output, mask)
                    val_loss += loss.item()

                    pred = torch.argmax(output, dim=1)
                    inter, union, d_num, d_den = compute_iou_and_dice(pred, mask)
                    for j in range(3):
                        total_inter[j] += inter[j]
                        total_union[j] += union[j]
                        total_dice_num[j] += d_num[j]
                        total_dice_den[j] += d_den[j]

                iou_vals = [total_inter[j] / (total_union[j] + 1e-6) for j in range(3)]
                dice_vals = [total_dice_num[j] / (total_dice_den[j] + 1e-6) for j in range(3)]

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1:2d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            print(f"   IoU:   {[round(i, 3) for i in iou_vals]}")
            print(f"   Dice:  {[round(d, 3) for d in dice_vals]}")

    print("\nSweep complete")
