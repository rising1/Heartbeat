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


        hawk_dataset = datasets.ImageFolder(root=self.train_path,
                                            transform=self.data_transform)
        print("hawk_dataset = ",len(hawk_dataset))
        train_loader = DataLoader(hawk_dataset, batch_size=self.batch_sizes,
                                  shuffle=True, num_workers=4)
        hawk_test_dataset = datasets.ImageFolder(root=self.test_path,
                                                 transform=self.data_transform_test)
        print("hawk_test_dataset = ", len(self.hawk_test_dataset))
        test_loader = DataLoader(hawk_test_dataset, batch_size=self.batch_sizes,
                                 shuffle=True, num_workers=4)
        # classes = ('sparrow hawk','red kite','peregrine falcon','kestrel','golden eagle','buzzard')
        classes = ('buzzard', 'golden eagle', 'kestrel', 'peregrine falcon',
                   'red kite', 'sparrow hawk')
        classesTest = ('buzzard', 'golden eagle', 'kestrel', 'peregrine falcon',
                       'red kite', 'sparrow hawk')
