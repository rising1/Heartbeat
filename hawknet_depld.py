import torch, torchvision
from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import DataLoader
import os, io
from matplotlib import pyplot as plt
import numpy as np

class test_images():

    def __init__(self,batch_size,shuffle_data):
        global data_transform
        self.batch_sizes = batch_size
        self.shuffle_data = shuffle_data
        data_transform = transforms.Compose([
                    transforms.Resize(128),
                    # transforms.CenterCrop(72),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def get_tensor(self,image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        return data_transform(image).unsqueeze(0)


    def data_transformation(self,image):
        global data_transform
        print("type(image)=",type(image))
        image_dataset = datasets.ImageFolder(image, data_transform)
        return image_dataset

    def eval_test(self,path_to_images):
        self.dir_path = path_to_images
        self.image_dataset = datasets.ImageFolder(self.dir_path+'/eval',data_transform)
        self.dataloaders = torch.utils.data.DataLoader(self.image_dataset,
                            batch_size=self.batch_sizes,
                            shuffle=self.shuffle_data,
                            num_workers=0)

        #print(type(self.dataloaders["train"][0]))
        self.dataset_sizes = len(self.image_dataset)
        return self.dataloaders

    def getitem(self, index):
        return self.image_dataset.__getitem__(index)   # return image path
