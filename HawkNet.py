import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
import ConvNet
import HawkDataLoader
from torch.optim import Adam
import time


# Hyperparameters
colour_channels = 3 # used in SimpleNet
no_feature_detectors = 12 # used in ??????
kernel_sizes = 3 # used in Unit
stride_pixels = 1 # used in Unit
padding_pixels = 1 # used in Unit
pooling_factor = 2 # used in SimpleNet
pic_size = 120 # used in SimpleNet
output_classes = 6 # used in SimpleNet
learning_rate = 0.0001 # used in HeartbeatClean
weight_decay = 0.0001 # used in HeartbeatClean
dropout_factor = 0.1 # used in Unit
snapshot_point = 20
faff = 'false'
dataPathRoot = 'C:/Users/phfro/Documents/python/data/BirdiesData/' # used in DataLoaderHeartbeat
num_epochs = 1 # used in HeartbeatClean
batch_sizes = 4 # used in HeartbeatClean

SimpleNetArgs = [kernel_sizes,stride_pixels,padding_pixels,dropout_factor,
                 output_classes,colour_channels,pic_size,pooling_factor]

cuda_avail = torch.cuda.is_available()
model = ConvNet.SimpleNet(SimpleNetArgs)
if cuda_avail:
    print("cuda is available")
    model.cuda()
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()

train_loader_class = HawkDataLoader.HawkLoader( \
                dataPathRoot,batch_sizes)

train_loader = train_loader_class.dataloaders["train"]

# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

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
        for (images, labels) in enumerate(train_loader):
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
        # adjust_learning_rate(epoch)
        # train_acc = train_acc.cpu().numpy() / len(hawk_dataset)
        # train_loss = train_loss / len(hawk_dataset)
        if ((epoch) % (num_epochs / snapshot_point) == 0) or (epoch == num_epochs):
            loopcount = loopcount + 1
            time_elapsed = time.time() - since
            print("Epoch {:4},Train Accuracy: {:.4f},TrainLoss: {:.4f}," \
                  .format(epoch, train_acc, \
                          train_loss), 'time {:.0f}m {:.0f}s'.format( \
                time_elapsed // 60, time_elapsed % 60))
            # print('Best val Acc: {:4f}'.format(best_acc))
            # Accuracy Curves
            train_history.append(train_acc)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


imshow(out, title=[x for x in train_loader_class.classes])

# train(num_epochs)
if __name__ == "__main__":

    train(num_epochs)