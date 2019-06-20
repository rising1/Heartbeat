import torch
import ConvNet
import os

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
model.load_state_dict(torch.load('Birdies_model_(90)_299.model',map_location='cpu'))
model.eval()

dataPathRoot = 'C:/Users/phfro/Documents/python/data/BirdiesData/' # used in DataLoaderHeartbeat
if not (os.path.exists(dataPathRoot)):
    dataPathRoot = 'C:/Users/peter.frost/Documents/python/data/BirdiesData/'  # used in DataLoaderHeartbeat
f = open(dataPathRoot + 'photo.jpg', 'r+')
jpgdata = f.read()
f.close()

#  Predict classes using images from the test set
outputs = model(jpgdata)
_, prediction = torch.max(outputs.data, 1)

