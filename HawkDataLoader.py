# transforms to apply to the datan
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import os


class HawkLoader:
    def __init__(self, dir_path, batch_sizes, pic_size):
        self.batch_sizes = batch_sizes
        self.dir_path = dir_path
        self.pic_size = pic_size
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.pic_size),
                #  transforms.CenterCrop(63),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(120),
                transforms.CenterCrop(self.pic_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(120),
                transforms.CenterCrop(self.pic_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # transforms.RandomResizedCrop(120,(1,1),(1,1),2),
        # print(os.path.join(self.dir_path, 'train'))
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.dir_path, x),
                                                  data_transforms[x])
                          for x in ['train', 'val', 'test']}
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
        self.classes = open('/content/drive/My Drive/Colab Notebooks/Class_validate.txt').read()
        print("self.classes=",self.classes)
        print("len self.classes=",len(self.classes))

