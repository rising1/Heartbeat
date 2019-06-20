import torch, torchvision
from torchvision import transforms, datasets
import ConvNet
import os
from matplotlib import pyplot as plt
import numpy as np
import PIL
from PIL import Image

# Hyperparameters
colour_channels = 3 # used in SimpleNet
no_feature_detectors = 12 # used in ??????
kernel_sizes = 3 # used in Unit
stride_pixels = 1 # used in Unit
padding_pixels = 1 # used in Unit
pooling_factor = 2 # used in SimpleNet
pic_size = 72 # used in SimpleNet
output_classes = 6 # used in SimpleNet
learning_rate = 0.0001 # used in HeartbeatClean
weight_decay = 0.0001 # used in HeartbeatClean
dropout_factor = 0.1 # used in Unit
faff = 'false'

SimpleNetArgs = [kernel_sizes, stride_pixels, padding_pixels, dropout_factor,
                 output_classes, colour_channels, pic_size, pooling_factor]

model = ConvNet.SimpleNet(SimpleNetArgs)
model.load_state_dict(torch.load('Birdies_model_299.model',map_location='cpu'))
model.eval()

def imshow(img):
    img = img  / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataPathRoot = 'C:/Users/phfro/Documents/python/data/BirdiesData/' # used in DataLoaderHeartbeat
if not (os.path.exists(dataPathRoot)):
    dataPathRoot = 'C:/Users/peter.frost/Documents/python/data/BirdiesData/'  # used in DataLoaderHeartbeat

data_transform = transforms.Compose([
                transforms.Resize(120),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image_dataset = datasets.ImageFolder(os.path.join(dataPathRoot, 'photo.jpg'), data_transform)
imshow(torchvision.utils.make_grid(image_dataset))

#  Predict classes using images from the test set
outputs = model(image_dataset)
_, prediction = torch.max(outputs.data, 1)

