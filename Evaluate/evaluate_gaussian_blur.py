import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pet_dataset import PetDataset
from unet import UNet
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

GAUSSIAN_KERNEL = torch.tensor([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 16.0

# Expand for 3 channels
GAUSSIAN_KERNEL = GAUSSIAN_KERNEL.repeat(3, 1, 1, 1)  


def apply_blur(img_tensor, times):
    mean = torch.tensor([0.4783, 0.4459, 0.3957]).view(3, 1, 1)
    std = torch.tensor([0.2261, 0.2230, 0.2247]).view(3, 1, 1)

    img = img_tensor.clone().cpu()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)

    img = img.unsqueeze(0) 
    for _ in range(times):
        img = F.conv2d(img, GAUSSIAN_KERNEL.to(img.device), padding=1, groups=3)
    img = img.squeeze(0)  

    img = (img - mean) / std
    return img.to(img_tensor.device)



def evaluate_on_gaussian_blur(data_path, model_path, device):
    blur_levels = list(range(10)) 
    test_dataset = PetDataset(data_path, test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    num_classes = 3
    mean_dice_scores = []

    with torch.no_grad():
        for blur_times in blur_levels:
            total_dice_num = [0.0] * num_classes
            total_dice_den = [0.0] * num_classes

            for img, gt_mask in test_loader:
                img = img.squeeze(1).to(device)         
                gt_mask = gt_mask.squeeze(1).to(device)  

                blurred_imgs = torch.stack([apply_blur(im, blur_times) for im in img])
                output = model(blurred_imgs)
                preds = torch.argmax(output, dim=1).cpu()

                for b in range(preds.shape[0]):
                    pred_b = preds[b]
                    gt_b = gt_mask[b].cpu()
                    valid_mask = (gt_b != 3)

                    masked_pred = pred_b[valid_mask]
                    masked_gt = gt_b[valid_mask]

                    for cls in range(num_classes):
                        pred_cls = (masked_pred == cls)
                        gt_cls = (masked_gt == cls)
                        intersection = (pred_cls & gt_cls).float().sum().item()
                        denom = pred_cls.float().sum().item() + gt_cls.float().sum().item()

                        total_dice_num[cls] += 2 * intersection
                        total_dice_den[cls] += denom

            avg_dice = [(total_dice_num[i] / (total_dice_den[i] + 1e-6)) for i in range(num_classes)]
            mean_dice = sum(avg_dice) / num_classes
            mean_dice_scores.append(mean_dice)
            print(f"Gaussian Blur {blur_times}x: Mean Dice = {mean_dice:.4f}")


    plt.figure(figsize=(8, 6))
    plt.plot(blur_levels, mean_dice_scores, marker='o')
    plt.xlabel("Number of Gaussian Blur Iterations")
    plt.ylabel("Mean Dice Score")
    plt.title("Segmentation Performance vs Gaussian Blur")
    plt.grid(True)
    plt.savefig("mean_dice_vs_gaussian_blur.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    DATA_PATH = "../Dataset"
    MODEL_PATH = "../Saved_Models/unet_weight2.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_on_gaussian_blur(DATA_PATH, MODEL_PATH, device)
