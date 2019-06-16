# Hyperparameters
colour_channels = 3
no_feature_detectors = 12
kernel_sizes = 3
stride_pixels = 1
padding_pixels = 1
pooling_factor = 2
pic_size = 120
output_classes = 6
learning_rate = 0.0001
weight_decay = 0.0001
batch_sizes = 64
dropout_factor = 0.1
snapshot_point = 20
faff = 'false'
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import time
from matplotlib import pyplot as plt
import matplotlib as mpl
import pathlib

# transforms to apply to the data
data_transform = transforms.Compose([
    # transforms.Resize(120),
    # transforms.RandomResizedCrop(120,(1,1),(1,1),2),
    transforms.RandomSizedCrop(120),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
data_transform_test = transforms.Compose([
    # transforms.Resize(120),
    transforms.RandomResizedCrop(120, (1, 1), (1, 1), 2),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
hawk_dataset = datasets.ImageFolder(root='C:/Users/phfro/Documents/python/data/BirdiesData/train/',
                                    transform=data_transform)
print("hawk_dataset = ", len(hawk_dataset))
train_loader = DataLoader(hawk_dataset,
                          batch_size=batch_sizes, shuffle=True, num_workers=4)
hawk_test_dataset = datasets.ImageFolder(root='C:/Users/phfro/Documents/python/data/BirdiesData/val/', \
                                         transform=data_transform_test)
print("hawk_test_dataset = ", len(hawk_test_dataset))
test_loader = DataLoader(hawk_test_dataset,
                         batch_size=batch_sizes, shuffle=True, num_workers=4)
# classes = ('sparrow hawk','red kite','peregrine falcon','kestrel','golden eagle','buzzard')
classes = ('buzzard', 'golden eagle', 'kestrel', 'peregrine falcon', 'red kite', 'sparrow hawk')
classesTest = ('buzzard', 'golden eagle', 'kestrel', 'peregrine falcon', 'red kite', 'sparrow hawk')


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels= \
            out_channels, kernel_size=kernel_sizes, \
                              stride=stride_pixels, padding=padding_pixels)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(p=dropout_factor)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        output = self.do(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self, num_classes=output_classes):
        super(SimpleNet, self).__init__()

        self.unit1 = Unit(in_channels=colour_channels, out_channels=pic_size)
        self.unit2 = Unit(in_channels=pic_size, out_channels=pic_size)
        self.unit3 = Unit(in_channels=pic_size, out_channels=pic_size)
        self.pool1 = nn.MaxPool2d(kernel_size=pooling_factor)

        self.unit4 = Unit(in_channels=pic_size, out_channels=pic_size * 2)
        self.unit5 = Unit(in_channels=pic_size * 2, out_channels=pic_size * 2)
        self.unit6 = Unit(in_channels=pic_size * 2, out_channels=pic_size * 2)
        self.unit7 = Unit(in_channels=pic_size * 2, out_channels=pic_size * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=pooling_factor)

        self.unit8 = Unit(in_channels=pic_size * 2, out_channels=pic_size * 4)
        self.unit9 = Unit(in_channels=pic_size * 4, out_channels=pic_size * 4)
        self.unit10 = Unit(in_channels=pic_size * 4, out_channels=pic_size * 4)
        self.unit11 = Unit(in_channels=pic_size * 4, out_channels=pic_size * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=pooling_factor)

        self.unit12 = Unit(in_channels=pic_size * 4, out_channels=pic_size * 4)
        self.unit13 = Unit(in_channels=pic_size * 4, out_channels=pic_size * 4)
        self.unit14 = Unit(in_channels=pic_size * 4, out_channels=pic_size * 4)
        self.avgpool = nn.AvgPool2d(kernel_size=(pooling_factor * 2) + 1)

        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, \
                                 self.pool1, self.unit4, self.unit5, self.unit6, \
                                 self.unit7, self.pool2, self.unit8, self.unit9, \
                                 self.unit10, self.unit11, self.pool3, \
                                 self.unit12, self.unit13, self.unit14, \
                                 self.avgpool)

        self.fc = nn.Linear(in_features=int(pic_size / (pooling_factor ** 3 * \
                                                        (pooling_factor * 2 + 1)) * \
                                            pic_size / (pooling_factor ** 3 * \
                                                        (pooling_factor * 2 + 1)) * \
                                            pic_size * 4), out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,
                             int(pic_size / (pooling_factor ** 3 * \
                                             (pooling_factor * 2 + 1)) * \
                                 pic_size / (pooling_factor ** 3 * \
                                             (pooling_factor * 2 + 1)) * \
                                 pic_size * 4))
        output = self.fc(output)
        return output


from torch.optim import Adam

cuda_avail = torch.cuda.is_available()
model = SimpleNet(num_classes=output_classes)
if cuda_avail:
    model.cuda()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()


# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):
    lr = learning_rate

    if epoch > 500:
        lr = lr / 2
    elif epoch > 350:
        lr = lr / 2
    elif epoch > 220:
        lr = lr / 2
    elif epoch > 180:
        lr = lr / 2
    elif epoch > 90:
        lr = lr / 2
    elif epoch > 60:
        lr = lr / 2
    elif epoch > 30:
        lr = lr / 2

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


path = pathlib.Path("C:/Users/phfro/Documents/python/data/BirdiesData/Heartbeat.pt")
print("path=", path)
print("path.exists=", path.exists())
# Function for a Yes/No result based on the answer provided as an arguement
open_model = "y"  # @param {type:"string"}


def yes_no(open_model):
    yes = set(['yes', 'y', 'ye', ''])
    no = set(['no', 'n'])

    while True:
        choice = (open_model).lower()
        if choice in yes:
            if (path.exists()):
                print("loading model from file ", path)
                model.load_state_dict(torch.load("C:/Users/phfro/Documents/python/data/BirdiesData/Heartbeat.pt"), \
                                      strict=False)
            else:
                print("using new model ")

            return True
        elif choice in no:
            return False
        else:
            print("Please respond with -yes or -no")


yes_no(open_model)


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if (faff == 'true'):
            imshow(torchvision.utils.make_grid(images[:4]))

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            if (faff == 'true'):
                print(' '.join('%5s' % classes[prediction[j]] for j in range(4)))
            test_acc += torch.sum(prediction == labels.data)
            if (faff == 'true'):
                print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    test_accT = test_acc.cpu().numpy() / len(hawk_test_dataset)
    return test_accT


def save_models(epoch):
    torch.save(model.state_dict(), "Birdies_model_{}.model".format(epoch))
    print("Chekcpoint saved")

def train(num_epochs):
    best_acc = 0.0
    since = time.time()
    train_history = []
    test_history = []
    loopcount = 0
    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            # optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("images.size(0)=",images.size(0))
            # print("loss.item().cpu().data[0]=",loss.item().cpu().data[0])
            train_loss += loss.cpu().item() * images.size(0)
            # train_loss += loss.cpu().data[0] * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data)
        adjust_learning_rate(epoch)
        train_acc = train_acc.cpu().numpy() / len(hawk_dataset)
        train_loss = train_loss / len(hawk_dataset)
        test_acc = test()
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc
        if ((epoch) % (num_epochs / snapshot_point) == 0) or (epoch == num_epochs):
            loopcount = loopcount + 1
            time_elapsed = time.time() - since
            print("Epoch {:4},Train Accuracy: {:.4f},TrainLoss: {:.4f},"
                  "Test Accuracy:{:.4f}".format(epoch, train_acc,
                                                train_loss, test_acc), 'time {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            # print('Best val Acc: {:4f}'.format(best_acc))
            # Accuracy Curves
            train_history.append(train_acc)
            test_history.append(test_acc)
            if (loopcount == (snapshot_point)):
                plt.figure(figsize=[8, 6])
                plt.plot(train_history, 'r', linewidth=3.0)
                plt.plot(test_history, 'b', linewidth=3.0)
                plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
                plt.xlabel('Epochs ', fontsize=16)
                plt.ylabel('Accuracy', fontsize=16)
                plt.title('Accuracy Curves', fontsize=16)
                plt.show()


if __name__ == "__main__":
    train(10)
