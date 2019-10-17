import torch, torchvision
from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import DataLoader
import os, io
from matplotlib import pyplot as plt
import numpy as np

class test_images():

    def __init__(self)
        data_transform = transforms.Compose([
                    transforms.Resize(80),
                    transforms.CenterCrop(72),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    global data_transform
    def get_tensor(self,image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        return data_transform(image).unsqueeze(0)


    def data_transformation(self,image):
        global data_transform
        print("type(image)=",type(image))
        #image_dataset = datasets.ImageFolder(os.path.join(dataPathRoot, 'photo.jpg'), data_transform)
        image_dataset = datasets.ImageFolder(image, data_transform)
        #  self.imshow(torchvision.utils.make_grid(image_dataset[0][0]))
        #  self.imshow(torchvision.utils.make_grid(image_dataset))
        #  push the data to the GPU
        #  image_dataset = image_dataset[0][0] --> replaced 27.08.19
        #  image_dataset = image_dataset[:][0]
        return image_dataset

    def eval_test(self,path_to_images):
        self.dir_path = path_to_images
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.dir_path, x),
                                                  data_transform[x])
                          for x in ['dummy']}
        self.dataloaders = {x: torch.utils.data.DataLoader(
                            image_datasets[x],
                            batch_size=self.batch_sizes,
                            shuffle=True, num_workers=0)
                       for x in ['dummy']}
        #print(type(self.dataloaders["train"][0]))
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['dummy']}
