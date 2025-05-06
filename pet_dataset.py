from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import torch
import random
from torchvision.transforms import InterpolationMode
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
class PetDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        self.test = test
        if test:
            self.images = sorted([root_path+"/Test/color/"+i for i in os.listdir(root_path+"/Test/color/")])
            self.masks = sorted([root_path+"/Test/label/"+i for i in os.listdir(root_path+"/Test/label/")])
        else:
            self.images = sorted([root_path+"/TrainVal/color/"+i for i in os.listdir(root_path+"/TrainVal/color/")])
            self.masks = sorted([root_path+"/TrainVal/label/"+i for i in os.listdir(root_path+"/TrainVal/label/")])
        
        # self.transform = transforms.Compose([
        #     transforms.Resize((256, 256), interpolation= InterpolationMode.BICUBIC),
        #     transforms.ToTensor()])
        # self.mask_transform = transforms.Compose([
        #     transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        #     transforms.ToTensor()])

        self.resize_img = transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC)
        self.resize_mask = transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.4783, 0.4459, 0.3957], std=[0.2261, 0.2230, 0.2247])


        self.train_aug = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.4),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(p=0.2),
        ])

    def __getitem__(self, index):
        seed = index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        img = self.resize_img(img)
        mask = self.resize_mask(mask)

        img_np = np.array(img)
        mask_np = np.array(mask)
        # img = self.transform(img)  
        # mask = self.mask_transform(mask) 
        
        # mask = (mask * 255).long() 

        if not self.test:
            # Apply augmentation during training only
            augmented = self.train_aug(image=img_np, mask=mask_np)
            img_np, mask_np = augmented["image"], augmented["mask"]

        img_tensor = self.to_tensor(img_np)
        mask_tensor = self.to_tensor(mask_np)
        img_tensor = self.normalize(img_tensor) 

        mask_tensor = (mask_tensor * 255).long()
        unique_vals = mask_tensor.unique()
        allowed = {0, 38, 75, 255}
        if not set(unique_vals.tolist()).issubset(allowed):
            print(f"Warning: Unexpected mask values found: {unique_vals.tolist()}")

        mask_tensor[(mask_tensor == 38)] = 1 
        mask_tensor[(mask_tensor == 75)] = 2
        mask_tensor[mask_tensor == 255] = 3
        
        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.images)
    

if __name__ == "__main__":
    DATA_PATH = "./Dataset" 

    train_dataset = PetDataset(root_path=DATA_PATH, test=False)
    test_dataset = PetDataset(root_path=DATA_PATH, test=True)

    def count_classes(dataset, name="Dataset"):
        cat_count = 0
        dog_count = 0
        for i in range(len(dataset)):
            _, mask_tensor = dataset[i]
            unique = torch.unique(mask_tensor)
            if 1 in unique:
                cat_count += 1
            if 2 in unique:
                dog_count += 1
        print(f"{name} Summary:")
        print(f"  Total Samples: {len(dataset)}")
        print(f"  Images with Cats (class 1): {cat_count}")
        print(f"  Images with Dogs (class 2): {dog_count}")
        print()

    count_classes(train_dataset, name="Train Dataset")
    count_classes(test_dataset, name="Test Dataset")
