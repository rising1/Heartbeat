import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from matplotlib import pyplot as plt
import numpy as np

class test_an_image():

    def __init__(self,dataPathRoot):

        self.dataPathRoot = dataPathRoot
        self.test_image = self.data_transformation(self.dataPathRoot)

    def imshow(self,img):
        img = img  / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def transfer_to_gpu(self,image_dataset):
        global device
        device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
        print("device=", device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            image_dataset[0][0].to(device)
            print("image transferred to GPU")
        return image_dataset[0][0]

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
        self.imshow(torchvision.utils.make_grid(image_dataset[0][0]))
        #  push the data to the GPU
        #  image_dataset = self.transfer_to_gpu(image_dataset)
        image_dataset = image_dataset[0][0]
        return image_dataset


