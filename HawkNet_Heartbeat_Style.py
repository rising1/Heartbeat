# Import needed packages
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import HawkDataLoader


class Unit(nn.Module):
    def __init__(self, argslist, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = argslist[0] # problem ?
        self.stride_pixels = argslist[1]
        self.padding_pixels = argslist[2]
        self.dropout_factor = argslist[3]

        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_sizes,
                              stride=self.stride_pixels,
                              padding=self.padding_pixels)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(p=self.dropout_factor)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        output = self.do(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self, SimpleNetArgs):

        super(SimpleNet, self).__init__()
        self.UnitArgs = [SimpleNetArgs[0], SimpleNetArgs[1], SimpleNetArgs[2], SimpleNetArgs[3]]
        self.num_classes = SimpleNetArgs[4]
        self.colour_channels = SimpleNetArgs[5]
        self.pic_size = SimpleNetArgs[6]
        self.pooling_factor = SimpleNetArgs[7]

        self.unit1 = UnitNet.Unit(self.UnitArgs, in_channels=self.colour_channels,
                                  out_channels=self.pic_size)
        self.unit2 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size,
                                  out_channels=self.pic_size)
        self.unit3 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size,
                                  out_channels=self.pic_size)
        self.pool1 = nn.MaxPool2d(kernel_size=self.pooling_factor)

        self.unit4 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size,
                                  out_channels=self.pic_size * 2)
        self.unit5 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 2,
                                  out_channels=self.pic_size * 2)
        self.unit6 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 2,
                                  out_channels=self.pic_size * 2)
        self.unit7 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 2,
                                 out_channels=self.pic_size * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=self.pooling_factor)

        self.unit8 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 2,
                                  out_channels=self.pic_size * 4)
        self.unit9 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
                                  out_channels=self.pic_size * 4)
        self.unit10 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
                                   out_channels=self.pic_size * 4)
        self.unit11 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
                                   out_channels=self.pic_size * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=self.pooling_factor)

        self.unit12 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
                                  out_channels=self.pic_size * 4)
        self.unit13 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
                                   out_channels=self.pic_size * 4)
        self.unit14 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
                                   out_channels=self.pic_size * 4)
        self.avgpool = nn.AvgPool2d(kernel_size=(self.pooling_factor * 2) + 1)

        self.net = nn.Sequential(self.unit1, self.unit2,
                                 self.unit3,
                                 self.pool1,
                                 self.unit4, self.unit5, self.unit6,
                                 self.unit7,
                                 self.pool2,
                                 self.unit8, self.unit9,
                                 self.unit10,
                                 self.unit11,
                                 self.pool3,
                                 self.unit12, self.unit13, self.unit14,
                                 self.avgpool)

        self.fc = nn.Linear(in_features=int(self.pic_size * 4), out_features=self.num_classes)


    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,int(self.pic_size * 4))
        output = self.fc(output)
        return output


batch_sizes = 72
pic_size = 72
dataPathRoot = 'C:/Users/phfro/PycharmProjects/Heartbeat'
validate_path = 'C:/Users/phfro/PycharmProjects/Heartbeat/Class_validate.txt'


train_loader_class = \
    HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
val_loader_class = \
    HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
test_loader_class = \
    HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
single_loader_class = \
    HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
train_loader = train_loader_class.dataloaders["train"]
# val_loader = val_loader_class.dataloaders["val"]
test_loader = test_loader_class.dataloaders["val"]





# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
model = SimpleNet(num_classes=220)

if cuda_avail:
    model.cuda()

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()


# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):
    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch):
    torch.save(model.state_dict(), "Birdies_Hearbeat_model_{}.model".format(epoch))
    print("Checkpoint saved")


def test():
    model.eval()
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        prediction = prediction.cpu().numpy()
        test_acc += torch.sum(prediction == labels.data)

    # Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / 10000

    return test_acc


def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().data[0] * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / 50000
        train_loss = train_loss / 50000

        # Evaluate on the test set
        test_acc = test()

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,
                                                                                        test_acc))


if __name__ == "__main__":
    train(200)
