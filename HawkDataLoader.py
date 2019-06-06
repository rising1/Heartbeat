# transforms to apply to the datan
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class HawkLoader:
    def __init__(self, dir_path, batch_sizes):
        self.batch_sizes = batch_sizes
        self.dir_path = dir_path
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # transforms.RandomResizedCrop(120,(1,1),(1,1),2),

        image_datasets = {x: datasets.ImageFolder(os.path.join(self.train_path, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                        batch_size=self.batch_sizes,
                                        shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


        classes = ('buzzard', 'golden eagle', 'kestrel', 'peregrine falcon',
                   'red kite', 'sparrow hawk')
        classesTest = ('buzzard', 'golden eagle', 'kestrel', 'peregrine falcon',
                       'red kite', 'sparrow hawk')
