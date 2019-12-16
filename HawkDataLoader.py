# transforms to apply to the datan
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch
import csv
import os


class HawkLoader:
    def __init__(self, dir_path, batch_sizes, pic_size, computer):
        self.batch_sizes = batch_sizes
        self.dir_path = dir_path
        self.pic_size = pic_size
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(80),
                # transforms.RandomResizedCrop(self.pic_size),
                transforms.CenterCrop(self.pic_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(80),
                transforms.CenterCrop(self.pic_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(80),
                transforms.CenterCrop(self.pic_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # transforms.RandomResizedCrop(120,(1,1),(1,1),2),
        # print(os.path.join(self.dir_path, 'train'))
        if (computer == "home_laptop" or computer == "home_red_room" ):
            image_datasets = {x: datasets.ImageFolder(os.path.join(self.dir_path, x),
                              data_transforms[x]) for x in ['train', 'val', 'test']}
        elif computer == "work":
            image_datasets = {x: datasets.ImageFolder(os.path.join('D:/', x),
                              data_transforms[x]) for x in ['train', 'val', 'test']}
        self.dataloaders = {x: torch.utils.data.DataLoader(
                            image_datasets[x],
                            batch_size=self.batch_sizes,
                            shuffle=True, num_workers=0)
                       for x in ['train', 'val', 'test']}
        #print(type(self.dataloaders["train"][0]))
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

        #  self.classes = open('BirdList.txt').read().splitlines()
        #  self.classesTest = ('buzzard', 'golden eagle','kestrel', 'peregrine falcon',
        #                    'red kite', 'sparrow hawk')

    def birds_listing(self):
        with open('/content/drive/My Drive/Colab Notebooks/Class_validate.txt', 'r') as f:
           reader = csv.reader(f)
           self.classes = list(reader)[0]
           self.classes.sort()
           #  self.classes = open('/content/drive/My Drive/Colab Notebooks/Class_validate.txt').read()
           #  print("self.classes=",self.classes)
           #  print("len self.classes=",len(self.classes))
        return self.classes

    def cifar10_train_loader(self):
        # Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
        train_transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load the training set
        train_set = CIFAR10(root="./data", train=True, transform=train_transformations, download=True)

        # Create a loder for the training set
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
        print ("returning train loader object")
        return train_loader

    def cifar10_test_loader(self):
        # Define transformations for the test set
        test_transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        ])

        # Load the test set, note that train is set to False
        test_set = CIFAR10(root="./data", train=False, transform=test_transformations, download=True)

        # Create a loder for the test set, note that both shuffle is set to false for the test loader
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
        return test_loader