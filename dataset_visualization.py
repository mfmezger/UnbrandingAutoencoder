import time
from DataSet import ImageDataSet
import numpy as np
import torch
from PIL import Image

dataset = ImageDataSet(root_dir="../../Desktop/data/")

x,y = dataset[0]

img = Image.fromarray(x)


img.show()


