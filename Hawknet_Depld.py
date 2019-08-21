import torch, torchvision
from torchvision import transforms, datasets
import ConvNet
import os
from matplotlib import pyplot as plt
import numpy as np
import PIL
from PIL import Image
import HawkNet

def imshow(img):
    img = img  / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataPathRoot = '/content/drive/My Drive/" \
                     "Colab Notebooks/eval/' # used in DataLoaderHeartbeat
if not (os.path.exists(dataPathRoot)):
    print(' data path doesnt exist')  # used in DataLoaderHeartbeat

data_transform = transforms.Compose([
                transforms.Resize(80),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#image_dataset = datasets.ImageFolder(os.path.join(dataPathRoot, 'photo.jpg'), data_transform)
image_dataset = datasets.ImageFolder(os.path.join(dataPathRoot), data_transform)
imshow(torchvision.utils.make_grid(image_dataset[0][0]))

#  Predict classes using images from the test set
outputs = HawkNet.model(image_dataset[0][0])
_, prediction = torch.max(outputs.data, 1)
print("prediction=",prediction)

