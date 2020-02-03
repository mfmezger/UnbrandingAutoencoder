import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFont
from PIL import ImageDraw
import copy


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

        image = Image.open(self.root_dir + img_name)
        image_gt = image.copy()

        # apply the text.

        img = Image.open("sample_in.jpg")

        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("sans-serif.ttf", 16)
        draw.text((0, 0), "ImageImageImage", (255, 255, 255), font=font)
        draw.text((200, 0), "ImageImageImage", (255, 255, 255), font=font)
        draw.text((450, 0), "ImageImageImage", (255, 255, 255), font=font)

        sample = {'image': image, 'gt': image_gt}

        if self.transform:
            sample = self.transform(sample)

        return sample