import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import shutil
import config


# class ImageDataset(Dataset):
#     def __init__(self, root_image, root_sketch):
#         self.transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.RandomHorizontalFlip()
#         ])
#         self.root_dir_image = root_image
#         self.root_dir_sketch = root_sketch
#         self.images_list = os.listdir(root_image)
#         self.sketches_list = os.listdir(root_sketch)
#
#     def __len__(self):
#         return len(self.images_list)
#
#     def __getitem__(self, index):
#         length = len(self.images_list) - len(self.sketches_list)
#         if config.PRINT_DATA_SHAPE:
#           print(f"distance: {length}")
#           print(f"Images length: {len(self.images_list)}")
#           print(f"Sketches length: {len(self.sketches_list)}")
#         assert length == 0, "Error: The images file and the sketches file must be at the same length"
#         img_file_sketch = self.sketches_list[index]
#         img_file_image = self.images_list[index]
#         img_path_sketch = os.path.join(self.root_dir_sketch, img_file_sketch)
#         img_path_image = os.path.join(self.root_dir_image, img_file_image)
#         sketch = Image.open(img_path_sketch)
#         image = Image.open(img_path_image)
#
#         if np.random.random() < 0.5:
#             sketch = Image.fromarray(np.array(sketch)[:, ::-1, :], "RGB")
#             image = Image.fromarray(np.array(image)[:, ::-1, :], "RGB")
#
#         sketch_ = self.transform(sketch)
#         image_ = self.transform(image)
#         x = transforms.ToTensor()
#         sketch_ = x(sketch_)
#         image_ = x(image_)
#         #print(f"Image size: {image_.shape}")
#         #print(f"Sketch size: {sketch_.shape}")
#         return sketch_, image_


def move_image(source_folder, destination_folder, image_name):
    source_path = os.path.join(source_folder, image_name)
    destination_path = os.path.join(destination_folder, image_name)
    shutil.move(source_path, destination_path)


def main():
    source = "./testDataHouseSketch"
    dest = "./valHouse1Sketch"
    for i in range(91, 232):
        name_p = f"{i}_sketch.jpg"
        move_image(source, dest, name_p)


if __name__ == '__main__':
    main()


class ImageDataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip()
        ])
        self.root = root
        self.images_list = os.listdir(root)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir_sketch, self.images_list[index])
        img = Image.open(img_path)
        width, height = img.size
        # 0 בציר ה- X וציר ה- Y, עד חצי ציר ה- X וסוף ציר ה- Y. צד שמאל
        image = img.crop((0, 0, width / 2, height))
        # צד ימין
        sketch = img.crop((width/2, 0, width, height))
        if np.random.random() < 0.5:
            sketch = Image.fromarray(np.array(sketch)[:, ::-1, :], "RGB")
            image = Image.fromarray(np.array(image)[:, ::-1, :], "RGB")

        sketch_ = self.transform(sketch)
        image_ = self.transform(image)
        trnsfrm = transforms.ToTensor()
        sketch_ = trnsfrm(sketch_)
        image_ = trnsfrm(image_)
        # print(f"Image size: {image_.shape}")
        # print(f"Sketch size: {sketch_.shape}")
        return sketch_, image_
