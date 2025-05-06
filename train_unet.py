import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import random
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from unet import UNet
from pet_dataset import PetDataset
import csv

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def dice_loss(pred, target, smooth=1e-6):
    num_classes = pred.shape[1]
    pred = torch.softmax(pred, dim=1)

    # Create mask to ignore class 3
    valid_mask = (target != 3) 

    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
    target_onehot = target_onehot * valid_mask.unsqueeze(1)
    pred = pred * valid_mask.unsqueeze(1)

    # Dice formula
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice_score = (2 * intersection + smooth) / (union + smooth)

    # Exclude class 3 from loss calculation
    valid_class_mask = torch.tensor([1, 1, 1, 0], dtype=torch.bool).to(pred.device)
    dice_score = dice_score[:, valid_class_mask]

    return 1 - dice_score.mean()


def combined_loss(pred, target, weight=None, ce_weight=1, dice_weight=1):
    ce_loss = F.cross_entropy(pred, target, weight=weight)
    d_loss = dice_loss(pred, target)
    print(f"CE Loss: {ce_loss.item():.4f}, Dice Loss: {d_loss.item():.4f}")
    return ce_weight * ce_loss + dice_weight * d_loss


def compute_iou_and_dice_cumulative(pred, target, num_classes=3):
    total_intersection = [0.0] * num_classes
    total_union = [0.0] * num_classes
    total_dice_num = [0.0] * num_classes
    total_dice_den = [0.0] * num_classes

    valid_mask = (target != 3)
    masked_pred = pred[valid_mask]
    masked_target = target[valid_mask]

    for cls in range(num_classes):
        pred_cls = (masked_pred == cls)
        target_cls = (masked_target == cls)

        intersection = (pred_cls & target_cls).float().sum().item()
        union = (pred_cls | target_cls).float().sum().item()
        dice_den = pred_cls.float().sum().item() + target_cls.float().sum().item()

        total_intersection[cls] += intersection
        total_union[cls] += union
        total_dice_num[cls] += 2 * intersection
        total_dice_den[cls] += dice_den

    return total_intersection, total_union, total_dice_num, total_dice_den



if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 8
    EPOCHS = 50                
    PATIENCE = 5               
    DATA_PATH = "./Dataset"
    MODEL_SAVE_PATH = "./Saved_Models/unet_weight2.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = PetDataset(DATA_PATH)

    total_samples = len(train_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=3, num_classes=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    #class_weights = torch.tensor([1.0, 3.0, 2.0, 0.5]).to(device)  # background, cat, dog, outline
    #criterion = nn.CrossEntropyLoss(weight=class_weights)
    #criterion = lambda pred, target: combined_loss(pred, target, alpha=0.3, beta=0.7, weight=class_weights)
    #criterion = lambda pred, target: combined_loss(pred, target, weight=class_weights)

    class_weights = torch.tensor([1.0, 2.0, 1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    METRICS_LOG_PATH = "./Saved_Models/metrics_log.csv"


    with open(METRICS_LOG_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        class_names = ['Background', 'Cat', 'Dog']
        header = ['Epoch', 'Train Loss', 'Val Loss'] + \
                [f'IoU_{cls}' for cls in class_names] + \
                [f'Dice_{cls}' for cls in class_names]
        writer.writerow(header)
    
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].long().to(device)
            mask = mask.squeeze(1)  
            
            optimizer.zero_grad()
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.empty_cache()
        
        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        total_iou = [0.0] * 3 
        total_dice = [0.0] * 3
        count = 0
        with torch.no_grad():
            total_intersection = [0.0] * 3
            total_union = [0.0] * 3
            total_dice_num = [0.0] * 3
            total_dice_den = [0.0] * 3
            val_running_loss = 0

            for idx, img_mask in enumerate(tqdm(val_dataloader, desc="Validation", leave=False)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].long().to(device)
                mask = mask.squeeze(1)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()

                pred = torch.argmax(y_pred, dim=1)  

                # Compute per-batch intersection, union, dice numerators and denominators
                for b in range(pred.shape[0]):
                    valid_mask = (mask[b] != 3)
                    pred_b = pred[b][valid_mask]
                    target_b = mask[b][valid_mask]

                    for cls in range(3): 
                        pred_cls = (pred_b == cls)
                        target_cls = (target_b == cls)

                        intersection = (pred_cls & target_cls).float().sum().item()
                        union = (pred_cls | target_cls).float().sum().item()
                        dice_den = pred_cls.float().sum().item() + target_cls.float().sum().item()

                        total_intersection[cls] += intersection
                        total_union[cls] += union
                        total_dice_num[cls] += 2 * intersection
                        total_dice_den[cls] += dice_den

                if device == "cuda":
                    torch.cuda.empty_cache()

            val_loss = val_running_loss / (idx + 1)
            avg_iou = [(total_intersection[i] / (total_union[i] + 1e-6)) for i in range(3)]
            avg_dice = [(total_dice_num[i] / (total_dice_den[i] + 1e-6)) for i in range(3)]

            scheduler.step(val_loss)
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 30)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print("Average IoU per class:")
        for cls, iou in zip(class_names, avg_iou):
            print(f"  {cls}: {iou:.4f}")

        print("Average Dice per class:") 
        for cls, dice in zip(class_names, avg_dice):
            print(f"  {cls}: {dice:.4f}")
        print("-" * 30)

        with open(METRICS_LOG_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch+1, train_loss, val_loss] + avg_iou + avg_dice
            writer.writerow(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Validation loss improved; saving model.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break
    print("Classweights: ", class_weights)
    print("Training complete.")
