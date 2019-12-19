# Import needed packages
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

# Hyper-parameters
colour_channels = 3  # used in SimpleNet
no_feature_detectors = 32  # used in ??????
kernel_sizes = 3  # used in Unit
stride_pixels = 1  # used in Unit
padding_pixels = 1  # used in Unit
pooling_factor = 2  # used in SimpleNet
pic_size = 32  # used in SimpleNet
output_classes = 10  # used in SimpleNet
learning_rate = 0.001  # used in HeartbeatClean
decay_cycles = 1  # default to start
weight_decay = 0.0001  # used in HeartbeatClean
dropout_factor = 0.4  # used in Unit
faff = 'false'
num_epochs = 20  # used in HeartbeatClean
snapshot_points = num_epochs / 1
batch_sizes = 128  # used in HeartbeatClean
#  batch_sizes = 6 # used in HeartbeatClean
loadfile = True
# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Args lists to pass through to models
UnitArgs = [kernel_sizes, stride_pixels, padding_pixels]
SimpleNetArgs = [UnitArgs, dropout_factor,output_classes, 
                 colour_channels, no_feature_detectors, 
                 pooling_factor]


class Unit(nn.Module):
    def __init__(self, UnitArgs, in_channels, out_channels):
        
        super(Unit, self).__init__()
        
        kernel_size = UnitArgs[0]
        stride = UnitArgs[1]
        padding = UnitArgs[2]

        self.conv = nn.Conv2d( kernel_size, stride, padding,
                               in_channels=in_channels, out_channels=out_channels)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output


class SimpleNet(nn.Module):
    def __init__(self, SimpleNetArgs):
        super(SimpleNet, self).__init__()

        # Break out the parameters for the model
        UnitArgs = SimpleNetArgs[0]
        dropout_factor = SimpleNetArgs[1]
        output_classes = SimpleNetArgs[2]
        colour_channels = SimpleNetArgs[3]
        no_feature_detectors = SimpleNetArgs[4]
        pooling_factor = SimpleNetArgs[5]


        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(UnitArgs,colour_channels, no_feature_detectors)
        self.unit2 = Unit(UnitArgs,no_feature_detectors, no_feature_detectors)
        self.unit3 = Unit(UnitArgs,no_feature_detectors, no_feature_detectors)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(UnitArgs,no_feature_detectors, no_feature_detectors * 2)
        self.unit5 = Unit(UnitArgs,no_feature_detectors * 2, no_feature_detectors * 2)
        self.unit6 = Unit(UnitArgs,no_feature_detectors * 2, no_feature_detectors * 2)
        self.unit7 = Unit(UnitArgs,no_feature_detectors * 2, no_feature_detectors * 2)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(UnitArgs,no_feature_detectors * 2, no_feature_detectors * 4)
        self.unit9 = Unit(UnitArgs,no_feature_detectors * 4, no_feature_detectors * 4)
        self.unit10 = Unit(UnitArgs,no_feature_detectors * 4, no_feature_detectors * 4)
        self.unit11 = Unit(UnitArgs,no_feature_detectors * 4, no_feature_detectors * 4)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(UnitArgs,no_feature_detectors * 4, no_feature_detectors * 4)
        self.unit13 = Unit(UnitArgs,no_feature_detectors * 4, no_feature_detectors * 4)
        self.unit14 = Unit(UnitArgs,no_feature_detectors * 4, no_feature_detectors * 4)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 , self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        return output


# Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = (128)

# Load the training set
train_set = CIFAR10(root="./data", train=True, transform=train_transformations, download=True)

# Create a loder for the training set
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# Define transformations for the test set
test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])

# Load the test set, note that train is set to False
test_set = CIFAR10(root="./data", train=False, transform=test_transformations, download=True)

# Create a loder for the test set, note that both shuffle is set to false for the test loader
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
model = SimpleNet(SimpleNetArgs)

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


def save_models(epoch,test_corrects):
    torch.save(model.state_dict(), "cifar10model" + test_corrects +"_{}.model".format(epoch))
    print("Checkpoint saved")


def test():
    global test_acc, test_acc_abs
    model.eval()
    test_acc_abs = 0
    test_acc = 0.0
    for i, (images, labels) in enumerate(test_loader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        # prediction = prediction.cpu()
        test_acc_abs += torch.sum(prediction == labels.data)

    # Compute the average acc and loss over all 10000 test images
    test_acc = test_acc_abs.cpu().numpy() / 10000
    return (test_acc, test_acc_abs)


def train(num_epochs):
    global best_acc, train_acc, train_loss
    best_acc = 0

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

            # train_loss += loss.cpu().data[0] * images.size(0)
            train_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Compute the average acc and loss over all 50000 training images
        train_acc = train_acc.cpu().numpy() / 50000
        train_loss = train_loss / 50000

        # Evaluate on the test set
        results = test()
        test_acc = results[0]
        test_acc_abs = results[1]

            # Save the model if the test acc is greater than our current best
        if test_acc_abs > best_acc and epoch%5 == 0 and epoch > 1:
                save_models(epoch,str(test_acc_abs.cpu().numpy()))
                best_acc = test_acc_abs

            # Print the metrics
        print("Epoch {}, Train Accuracy: {:.1%} , TrainLoss: {:.4f} , Test Accuracy: {:.1%}, Test Accuracy Absolute: {}".format(epoch, train_acc, train_loss,
                                                                                        test_acc, test_acc_abs))


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Changed num_workers to 0, fixed prediction == labels.data,
    #-------------------------------------------------------------------
    train(50)
