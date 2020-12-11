# G# Imports here
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Load resnet-50 pre-trained network
model = models.resnet50(pretrained=True)
print(model)
