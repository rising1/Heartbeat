# transforms to apply to the datan
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
import os

class HawkLoader:
    def __init__(self, dir_path, batch_sizes):
        self.batch_sizes = batch_sizes
        self.dir_path = dir_path
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(48),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(120),
                transforms.CenterCrop(48),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # transforms.RandomResizedCrop(120,(1,1),(1,1),2),
        # print(os.path.join(self.dir_path, 'train'))
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.dir_path, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(
                            image_datasets[x],
                            batch_size=self.batch_sizes,
                            shuffle=True, num_workers=0)
                       for x in ['train', 'val']}
        #print(type(self.dataloaders["train"][0]))
        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        self.classes = ('buzzard', 'golden eagle', 'harris hawk', 'kestrel', 'peregrine falcon',
                        'red kite', 'sparrow hawk')
        self.classesTest = ('buzzard', 'golden eagle', 'harris hawk','kestrel', 'peregrine falcon',
                            'red kite', 'sparrow hawk')
