import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

from pet_dataset import PetDataset
from unet import UNet

import numpy as np
PALETTE = {
    0: (0, 0, 0),        # Black for background
    1: (255, 0, 0),      # Red for cat
    2: (0, 255, 0),      # Green for dog
    3: (255, 255, 255)   # White for ignored region (if used visually)
}
def colorize_mask(mask, palette):
    mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in palette.items():
        color_mask[mask == cls] = color
    return color_mask

def compute_multiclass_iou_and_dice(pred, target, num_classes=3):
    eps = 1e-6
    iou_list = []
    dice_list = []
    valid_mask = (target != 3)
    masked_pred = pred[valid_mask]
    masked_target = target[valid_mask]

    for cls in range(num_classes):
        pred_cls = (masked_pred == cls)
        target_cls = (masked_target == cls)

        intersection = (pred_cls & target_cls).float().sum()
        union = pred_cls.float().sum() + target_cls.float().sum() - intersection
        iou = intersection / (union + eps)
        dice = (2 * intersection) / (pred_cls.float().sum() + target_cls.float().sum() + eps)

        iou_list.append(iou.item())
        dice_list.append(dice.item())

    return iou_list, dice_list

def pred_show_image_grid(data_path, model_pth, device):
    # Instantiate the model with 4 output channels (0=bg, 1=cat, 2=dog, 3=ignore)
    model = UNet(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    model.eval()
    
    image_dataset = PetDataset(data_path, test=True)
    num_samples = len(image_dataset)

    images = []
    orig_masks = []
    pred_masks = []
    sample_mean_iou = []
    sample_mean_dice = []

    num_classes = 3  # background, cat, dog

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img_batch = img.unsqueeze(0)  

        with torch.no_grad():
            output = model(img_batch) 
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()

        img_disp = img.cpu().permute(1, 2, 0) 
        orig_mask = orig_mask.squeeze(0).cpu()

        images.append(img_disp)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

        iou_list, dice_list = compute_multiclass_iou_and_dice(pred_mask, orig_mask, num_classes=num_classes)
        mean_iou = sum(iou_list) / num_classes
        mean_dice = sum(dice_list) / num_classes
        sample_mean_iou.append(mean_iou)
        sample_mean_dice.append(mean_dice)

        torch.cuda.empty_cache()

    fig = plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        ax1 = fig.add_subplot(4, num_samples, i + 1)
        ax1.imshow(images[i].numpy())
        ax1.set_title("Image")
        ax1.axis("off")

        ax2 = fig.add_subplot(4, num_samples, num_samples + i + 1)
        colored_gt = colorize_mask(orig_masks[i], PALETTE)
        ax2.imshow(colored_gt)
        ax2.set_title("Orig Mask")
        ax2.axis("off")

        ax3 = fig.add_subplot(4, num_samples, 2 * num_samples + i + 1)
        colored_pred = colorize_mask(pred_masks[i], PALETTE)
        ax3.imshow(colored_pred)
        ax3.set_title("Pred Mask")
        ax3.axis("off")

        ax4 = fig.add_subplot(4, num_samples, 3 * num_samples + i + 1)
        ax4.text(0.5, 0.5, f"mIoU: {sample_mean_iou[i]:.2f}\nmDice: {sample_mean_dice[i]:.2f}",
                 fontsize=12, ha='center', va='center')
        ax4.axis("off")

    plt.tight_layout()
    plt.savefig("prediction_grid.png", dpi=300, bbox_inches="tight")
    plt.close()


def single_image_inference(data_path, model_pth, device, target_filename):
    model = UNet(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    model.eval()

    dataset = PetDataset(data_path, test=True)
    idx = [i for i, path in enumerate(dataset.images) if os.path.basename(path) == target_filename]
    if not idx:
        print(f"Image {target_filename} not found in dataset.")
        return
    img, gt_mask = dataset[idx[0]]

    img_tensor = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(img_tensor), dim=1).squeeze(0).cpu()

    gt_mask = gt_mask.squeeze(0).cpu()
    img_disp = img.cpu().permute(1, 2, 0).numpy()
    img_disp = np.clip((img_disp * np.array([0.2261, 0.2230, 0.2247]) + np.array([0.4783, 0.4459, 0.3957])), 0, 1)

    iou_list, dice_list = compute_multiclass_iou_and_dice(pred, gt_mask, num_classes=3)
    present = torch.unique(gt_mask)
    present = present[present != 3]
    mean_iou = np.mean([iou_list[c] for c in present])
    mean_dice = np.mean([dice_list[c] for c in present])

    metrics = [f"{['Background','Cat','Dog'][c]} - IoU: {iou_list[c]:.2f}, Dice: {dice_list[c]:.2f}" for c in present]
    metrics.append(f"\nMean IoU: {mean_iou:.2f}, Mean Dice: {mean_dice:.2f}")
    metrics_text = "\n".join(metrics)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_disp)
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(colorize_mask(gt_mask, PALETTE))
    ax2.set_title("Ground Truth Mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(colorize_mask(pred, PALETTE))
    ax3.set_title("Predicted Mask")
    ax3.axis("off")
    ax3.text(0.5, -0.1, metrics_text, fontsize=10, ha='center', va='center', transform=ax3.transAxes)

    plt.tight_layout()
    plt.show()


def evaluate_test_dataset(data_path, model_pth, device):
    model = UNet(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    model.eval()
    
    test_dataset = PetDataset(data_path, test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    num_classes = 3
    total_intersection = [0.0] * num_classes
    total_union = [0.0] * num_classes
    total_dice_num = [0.0] * num_classes
    total_dice_den = [0.0] * num_classes

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.float().to(device)
            masks = masks.long().squeeze(1).to(device)  

            output = model(images)                      
            pred = torch.argmax(output, dim=1)       

            for b in range(pred.shape[0]):
                valid_mask = (masks[b] != 3)
                pred_b = pred[b][valid_mask]
                target_b = masks[b][valid_mask]

                for cls in range(num_classes):
                    pred_cls = (pred_b == cls)
                    target_cls = (target_b == cls)

                    intersection = (pred_cls & target_cls).float().sum().item()
                    union = (pred_cls | target_cls).float().sum().item()
                    dice_den = pred_cls.float().sum().item() + target_cls.float().sum().item()

                    total_intersection[cls] += intersection
                    total_union[cls] += union
                    total_dice_num[cls] += 2 * intersection
                    total_dice_den[cls] += dice_den

            torch.cuda.empty_cache()

    avg_iou = [(total_intersection[i] / (total_union[i] + 1e-6)) for i in range(num_classes)]
    avg_dice = [(total_dice_num[i] / (total_dice_den[i] + 1e-6)) for i in range(num_classes)]
    mean_iou = sum(avg_iou) / num_classes
    mean_dice = sum(avg_dice) / num_classes

    return {
        "avg_iou": avg_iou,
        "avg_dice": avg_dice,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice
    }



if __name__ == "__main__":
    TARGET_FILENAME = "english_setter_96.jpg"
    DATA_PATH = "./Dataset"
    MODEL_PATH = "./Saved_Models/unet_weight2.pth"
    # Class IDs: 0 - Background, 1 - Cat, 2 - Dog, 3 - Border (ignored)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    single_image_inference(DATA_PATH, MODEL_PATH, device="cpu", target_filename=TARGET_FILENAME)
    
    # Evaluate the entire test dataset without display.
    metrics = evaluate_test_dataset(DATA_PATH, MODEL_PATH, device)
    class_names = ['Background', 'Cat', 'Dog']

    print("Test Dataset Metrics:")
    print("Average IoU per class:")
    for name, val in zip(class_names, metrics["avg_iou"]):
        print(f"  {name}: {val:.4f}")

    print("Average Dice per class:")
    for name, val in zip(class_names, metrics["avg_dice"]):
        print(f"  {name}: {val:.4f}")

    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Dice: {metrics['mean_dice']:.4f}")

