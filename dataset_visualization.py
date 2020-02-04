import time
from DataSet import ImageDataSet
import numpy as np
import torch
from PIL import Image

dataset = ImageDataSet(root_dir="../../Desktop/data/", patchsize=300)
x,y = dataset[0]
img = Image.fromarray(x)
img_gt = Image.fromarray(y)
img.show()
img_gt.show()


