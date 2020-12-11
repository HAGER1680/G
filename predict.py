# G# Imports here
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

def predict(image_path, model, topk=3, transform):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = transform(Image.open(image_path)).unsqueeze(0)
    model.eval()
    output = model.forward(Variable(image))
    probabilities = torch.exp(output).data.numpy()[0]
    

    top_idx = np.argsort(probabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = probabilities[top_idx]

    return top_idx , top_probability, top_class
