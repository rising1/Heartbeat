import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
import ConvNet
import HawkDataLoader
from torch.optim import Adam
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import time


# Hyperparameters
colour_channels = 3 # used in SimpleNet
no_feature_detectors = 12 # used in ??????
kernel_sizes = 3 # used in Unit
stride_pixels = 1 # used in Unit
padding_pixels = 1 # used in Unit
pooling_factor = 2 # used in SimpleNet
pic_size = 72 # used in SimpleNet
output_classes = 220 # used in SimpleNet
learning_rate = 0.0001 # used in HeartbeatClean
weight_decay = 0.0001 # used in HeartbeatClean
dropout_factor = 0.1 # used in Unit
faff = 'false'
#dataPathRoot = 'F:/BirdiesData/' # used in DataLoaderHeartbeat
#dataPathRoot = 'C:/Users/phfro/Documents/python/data/BirdiesData/' # used in DataLoaderHeartbeat
#if not (os.path.exists(dataPathRoot)):
#    dataPathRoot = 'C:/Users/peter.frost/Documents/python/data/BirdiesData/'  # used in DataLoaderHeartbeat
#  dataPathRoot = '/content/drive/My Drive/Colab Notebooks/BirdiesData'
dataPathRoot = '/content/drive/My Drive/Colab Notebooks'
num_epochs = 300 # used in HeartbeatClean
snapshot_points = num_epochs / 1
batch_sizes = 64 # used in HeartbeatClean
#  batch_sizes = 6 # used in HeartbeatClean
print("parameters loaded")

SimpleNetArgs = [kernel_sizes, stride_pixels, padding_pixels, dropout_factor,
                 output_classes, colour_channels, pic_size, pooling_factor]
model = ConvNet.SimpleNet(SimpleNetArgs)
#if  (os.path.exists('Birdies_model_299.model')):
#    model.load_state_dict(torch.load('Birdies_model_299.model', map_location='cpu'))
#    print("using saved model")
#else:
#    print("using new model")

optimizer = Adam(model.parameters(), lr=learning_rate,
                 weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
print("device=",device)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model.to(device)

train_loader_class = \
        HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
val_loader_class = \
        HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
test_loader_class = \
        HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
single_loader_class = \
        HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
train_loader = train_loader_class.dataloaders["train"]
val_loader = val_loader_class.dataloaders["val"]
test_loader = test_loader_class.dataloaders["test"]


# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)


def save_models(epoch):
    torch.save(model.state_dict(), "Birdies_model_{}.model".format(epoch))
    print("Chekcpoint saved")


def train(num_epochs):
    print("In train")
    best_acc = 0.0
    since = time.time()
    train_history = []
    loopcount = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #  print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
                print('model set to train mode')
            else:
                model.eval()   # Set model to evaluate mode
                print('model set to eval mode')

            running_loss = 0.0
            running_corrects = 0
            counter = 0
            # Iterate over data.
            for inputs, labels in train_loader_class.dataloaders[phase]:
                print("inputs size=",inputs.shape)
                print("labels size=",labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            train_loss = running_loss / train_loader_class.dataset_sizes[phase]
            train_acc = running_corrects.double() / \
                        train_loader_class.dataset_sizes[phase]

            # Evaluate on the test set
            test_acc = test_train()

            # Save the model if the test acc is greater than our current best
            if test_acc > best_acc:
                save_models(epoch)
                best_acc = test_acc
                print("best accuracy= ", best_acc)

        if ((epoch) % (num_epochs / snapshot_points) == 0) or (epoch == num_epochs):
            loopcount = loopcount + 1
            time_elapsed = time.time() - since
            print("Epoch {:4}, ".format(epoch),
                 phase, " Accuracy: {:.4f},TrainLoss: {:.4f},"
                  .format( train_acc,
                          train_loss), 'time {:.0f}m {:.0f}s'.format(
                          time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))
            # Accuracy Curves
            train_history.append(train_acc)


def test_train():
    model.eval()
    test_acct = 0.0
    test_history = []
    images, labels = next(iter(test_loader))

    if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

    #  Predict classes using images from the test set
    outputs = model(images)
    _, prediction = torch.max(outputs.data, 1)

    test_acct += torch.sum(prediction == labels.data)
    #  print("test_acct= ",  (test_acct).cpu().numpy())
    #  Compute the average acc and loss over all 10000 test images
    test_acct = test_acct.cpu().numpy() / 30
    test_history.append(test_acct)
    #  print("in test")
    return test_acct

def test():
    model.eval()
    test_acct = 0.0
    test_history = []
    images, labels = next(iter(test_loader))

    if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

    #  Predict classes using images from the test set
    outputs = model(images)
    _, prediction = torch.max(outputs.data, 1)
    print("prediction=",single_loader_class.classes[int(prediction.cpu().numpy())])


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
    plt.pause(0.1)  # pause a bit so that plots are updated


#imshow(out, title=[x for x in train_loader_class.classes])

# train(num_epochs)
if __name__ == "__main__":
      train(num_epochs)
    #test()
