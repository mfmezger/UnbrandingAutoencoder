import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFont
import numpy as np
from PIL import ImageDraw


class ImageDataSet(Dataset):
    """Image dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_dir[idx]

        image = Image.open(os.path.join(self.root_dir + img_name))
        image_gt = image.copy()

        # apply the text.

        # Randomize text here.
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("Arial.ttf", 16)
        width, height = image.size

        x = 0
        while x  < width:
            y = 0
            while y < height:
                draw.text((x, y), "Image", (255, 255, 255), font=font)
                y += 100
            x += 100


        image = np.array(image)
        image_gt = np.array(image_gt)
        #sample = {'image': image, 'gt': image_gt}

        #if self.transform:
        #    sample = self.transform(sample)

        return image, image_gt