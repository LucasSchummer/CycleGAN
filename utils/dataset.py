from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random

class UnpairedDataset(Dataset):

    def __init__(self, root_dir, mode="train", img_size=128):

        self.dir_A = os.path.join(root_dir, f"{mode}A")
        self.dir_B = os.path.join(root_dir, f"{mode}B")

        self.A_images = sorted(os.listdir(self.dir_A))
        self.B_images = sorted(os.listdir(self.dir_B))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1] range
        ])

    def __len__(self):

        return max(len(self.A_images), len(self.B_images))

    def __getitem__(self, idx):

        img_A = Image.open(os.path.join(self.dir_A, self.A_images[idx % len(self.A_images)])).convert("RGB")
        img_B = Image.open(os.path.join(self.dir_B, random.choice(self.B_images))).convert("RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}
    
