import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from torch.utils.data import DataLoader

from pet_dataset import PetDataset
from unet import UNet

def compute_dice(pred, target, num_classes=3):
    eps = 1e-6
    dice_list = []
    valid_mask = (target != 3)
    pred = pred * valid_mask
    target = target * valid_mask
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        dice = (2 * intersection) / (pred_cls.sum() + target_cls.sum() + eps)
        dice_list.append(dice.item())
    return dice_list


def add_salt_pepper_noise(img_tensor, amount):
    mean = [0.4783, 0.4459, 0.3957]
    stds = [0.2261, 0.2230, 0.2247]


    unnorm = img_tensor.clone()
    for c in range(3):
        unnorm[c] = unnorm[c] * stds[c] + mean[c]
    unnorm = torch.clamp(unnorm, 0, 1)

    np_img = unnorm.permute(1, 2, 0).numpy()

    noisy = random_noise(np_img, mode='s&p', amount=amount)
    noisy = np.clip(noisy, 0, 1)

    noisy_tensor = torch.from_numpy(noisy).permute(2, 0, 1).float()
    for c in range(3):
        noisy_tensor[c] = (noisy_tensor[c] - mean[c]) / stds[c]

    return noisy_tensor


def evaluate_on_salt_pepper_noise(data_path, model_path, device):
    noise_amounts = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]

    test_dataset = PetDataset(data_path, test=True)
    model = UNet(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    mean_dice_scores = []
    example_results = {}
    num_classes = 3

    for amount in noise_amounts:
        total_dice_num = [0.0] * num_classes
        total_dice_den = [0.0] * num_classes
        first_example_stored = False

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for img, gt_mask in test_loader:
                img = img.squeeze(1)
                gt_mask = gt_mask.squeeze(1)

                perturbed_imgs = torch.stack([add_salt_pepper_noise(im, amount) for im in img])
                perturbed_imgs = perturbed_imgs.to(device)

                output = model(perturbed_imgs)
                preds = torch.argmax(output, dim=1).cpu()

                for b in range(preds.shape[0]):
                    pred_b = preds[b]
                    gt_b = gt_mask[b]
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

                    if not first_example_stored:
                        example_results[amount] = {
                            "original": img[b].cpu().numpy().transpose(1, 2, 0),
                            "perturbed": perturbed_imgs[b].cpu().numpy().transpose(1, 2, 0),
                            "pred_mask": pred_b.numpy(),
                            "gt_mask": gt_b.numpy()
                        }
                        first_example_stored = True

        avg_dice = [(total_dice_num[i] / (total_dice_den[i] + 1e-6)) for i in range(num_classes)]
        mean_dice = sum(avg_dice) / num_classes
        mean_dice_scores.append(mean_dice)
        print(f"Salt & Pepper Amount {amount:.2f}: Mean Dice = {mean_dice:.4f}")

    # Save Plot
    plt.figure(figsize=(8, 6))
    plt.plot(noise_amounts, mean_dice_scores, marker='o')
    plt.xlabel("Salt & Pepper Noise Amount")
    plt.ylabel("Mean Dice Score")
    plt.title("Segmentation Performance vs Salt & Pepper Noise")
    plt.grid(True)
    plt.savefig("mean_dice_vs_salt_pepper.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    DATA_PATH = "../Dataset"
    MODEL_PATH = "../Saved_Models/unet_weight2.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_on_salt_pepper_noise(DATA_PATH, MODEL_PATH, device)

