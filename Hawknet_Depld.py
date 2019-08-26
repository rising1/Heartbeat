import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
import numpy as np

class test_images():

    def __init__(self,dataPathRoot):

        self.dataPathRoot = dataPathRoot
        self.data_loaded = self.data_loader(self.data_transformation(
                            self.dataPathRoot))

    def data_transformation(self,dataPathRoot):
        if not (os.path.exists(dataPathRoot)):
            print(' data path doesnt exist')  # used in DataLoaderHeartbeat

        data_transform = transforms.Compose([
                transforms.Resize(80),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        #image_dataset = datasets.ImageFolder(os.path.join(dataPathRoot, 'photo.jpg'), data_transform)
        image_dataset = datasets.ImageFolder(os.path.join(dataPathRoot), data_transform)
        #  self.imshow(torchvision.utils.make_grid(image_dataset[0][0]))
        #  self.imshow(torchvision.utils.make_grid(image_dataset))
        #  push the data to the GPU
        #  image_dataset = image_dataset[0][0]
        return image_dataset

    def data_loader(self,image_dataset):
        return DataLoader(image_dataset,batch_size=6,shuffle=False,num_workers=0)


