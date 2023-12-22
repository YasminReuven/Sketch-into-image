import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip()
        ])
        self.root_dir = root
        self.files_list = os.listdir(root)

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, item):
        img_file = self.files_list[item]
        img_path = os.path.join(self.root_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        img = np.array(img)

