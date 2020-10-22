import os
import torch
import cv2
import numpy as np
from functools import partial

dir_path = os.path.dirname(os.path.realpath(__file__))
colors_path = os.path.join(dir_path, "resource", "colors.th")
mean_color = torch.load(colors_path)

def colorize(raw_image, severity, attribute_label):    
    std_color = [0.05, 0.02, 0.01, 0.005, 0.002][severity-1]
    image = (
        torch.clamp(mean_color[attribute_label]
        + torch.randn((3, 1, 1)) * std_color, 0.0, 1.0)
    ) * raw_image.unsqueeze(0).float()
    
    return image

COLORED_MNIST_PROTOCOL = dict()
for i in range(10):
    COLORED_MNIST_PROTOCOL[i] = partial(colorize, attribute_label = i)