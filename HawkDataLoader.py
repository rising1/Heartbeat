# transforms to apply to the datan
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class HawkLoader:
    def __init__(self, train_path, test_path, batch_sizes):
        self.batch_sizes = batch_sizes
        self.train_path = train_path
        self.test_path = test_path
        self.data_transform = transforms.Compose([
            # transforms.Resize(120),
            # transforms.RandomResizedCrop(120,(1,1),(1,1),2),
            transforms.RandomSizedCrop(120),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
            ])
        self.data_transform_test = transforms.Compose([
            # transforms.Resize(120),
            transforms.RandomResizedCrop(120, (1, 1), (1, 1), 2),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
            ])
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
