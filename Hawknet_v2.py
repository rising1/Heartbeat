# Import needed packages
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import time
import os, csv
import glob
import Hawknet_Depld
import View_Test
import HawkDataLoader

# Hyper-parameters
colour_channels = 3  # used in SimpleNet
no_feature_detectors = 64 # used in Unit
kernel_sizes = 3  # used in Unit
stride_pixels = 1  # used in Unit
padding_pixels = 1  # used in Unit
pooling_factor = 2  # used in SimpleNet
pic_size = 128 # used in SimpleNet
output_classes = 220  # used in SimpleNet
learning_rate = 0.001  # used in HeartbeatClean
decay_cycles = 1  # default to start
weight_decay = 0.0001  # used in HeartbeatClean
dropout_factor = 0.2  # used in Unit
faff = 'false'
num_epochs = 200  # used in HeartbeatClean
snapshot_points = num_epochs / 1
batch_sizes = 32 # used in HeartbeatClean
#  batch_sizes = 6 # used in HeartbeatClean
loadfile = True

validate_path = '/content/drive/My Drive/Colab Notebooks/Class_validate.txt'
dataPathRoot = '/content/drive/My Drive/Colab Notebooks'
# dataPathRoot = 'C:/Users/phfro/PycharmProjects/Heartbeat'
# dataPathRoot = 'E:/'
# validate_path = 'C:/Users/phfro/PycharmProjects/Heartbeat/Class_validate.txt'
# validate_path = 'E:/Class_validate.txt'
computer = "home_laptop"
deploy_test = Hawknet_Depld.test_images(12, False)
# Check if gpu support is available
cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Args lists to pass through to models
UnitArgs = [kernel_sizes, stride_pixels, padding_pixels]
SimpleNetArgs = [UnitArgs, dropout_factor,output_classes, 
                 colour_channels, no_feature_detectors, 
                 pooling_factor]


class Unit(nn.Module):
    def __init__(self, UnitArgs, in_channel, out_channel):
        
        super(Unit, self).__init__()
        self.conv = nn.Conv2d( kernel_size = UnitArgs[0], stride = UnitArgs[1],
                               padding = UnitArgs[2],
                               in_channels = in_channel, out_channels = out_channel)
        self.bn = nn.BatchNorm2d(num_features=out_channel)
        # self.do = nn.Dropout(dropout_factor)
        self.relu = nn.ReLU()


    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        # output = self.do(output)
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

        self.fc = nn.Linear(no_feature_detectors * 4 * 4, output_classes)

    def forward(self, input):
        output = self.net(input)
        #print("net(input) ",output.shape)
        output = output.view(-1, no_feature_detectors * 4 * 4)
        #print("output.view ",output.shape)
        output = self.fc(output)
        #print("fc(output) ",output.shape)
        return output


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def first_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("learning rate adjusted to ", lr)


def lr_decay_cycles(cycles):
    global decay_cycles
    decay_cycles = cycles
    return decay_cycles


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every decay epochs"""
    global decay_cycles
    learning_rate = get_lr(optimizer) * (0.1 ** (epoch // decay_cycles))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def get_latest_file(path, *paths):
    """Returns the name of the latest (most recent) file
    of the joined path(s)"""
    fullpath = os.path.join(path, *paths)
    list_of_files = glob.glob(fullpath)  # You may use iglob in Python3
    if not list_of_files:                # I prefer using the negation
        return None                      # because it behaves like a shortcut
    latest_file = max(list_of_files, key=os.path.getctime)
    _, filename = os.path.split(latest_file)
    return filename

def load_latest_saved_model(chosen_model = None,is_eval = False):
    global dataPathRoot, loadfile, model, optimizer, \
            epoch, loss, device
    # load a saved model if one exists
    comp_root = dataPathRoot + "/saved_models/"

    if chosen_model is not None:
        selected_model = chosen_model
        print("looking for ",comp_root + selected_model)
        print("exists = ",os.path.isfile(comp_root + selected_model))
    else:
        stub_name = "Birdies_model_*"
        selected_model = get_latest_file(comp_root, stub_name)
        print("latest filename=", selected_model)

    if os.path.isfile(comp_root + selected_model) and loadfile == True:
        checkpoint = torch.load(comp_root +  selected_model,map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        #  model.train()
        if not is_eval:
            model_file_path = comp_root + selected_model
            interim_fig_prev_text = model_file_path[(model_file_path.rfind('_') + 1):(len(model_file_path) - 6)]
            interim_fig_prev = float(interim_fig_prev_text)
            print("using saved model ", model_file_path, " Loss: {:.4f}".format(interim_fig_prev))
    else:
        print("using new model")
    #  finished deciding where the model comes from

    #  For the given model

    #  Print model's state_dict
    #  print("Model's state_dict:")
    #  for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        if var_name == "param_groups":
            print(var_name, "\t", optimizer.state_dict()[var_name])
    first_learning_rate(optimizer,learning_rate)
    print("model loaded")

batch_size = batch_sizes

#Load the training set
#train_set = CIFAR10(root="./data", train=True, #transform=train_transformations, download=True)
#Create a loder for the training set
#train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, #num_workers=4)

#Define transformations for the test set
#test_transformations = transforms.Compose([
# transforms.ToTensor(),
# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#])
# Load the test set, note that train is set to False
#test_set = CIFAR10(root="./data", train=False, transform=test_transformations, #download=True)

# Create a loder for the test set, note that both shuffle is set to false for #the test loader
#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, #num_workers=4)

loader = HawkDataLoader.HawkLoader(dataPathRoot,batch_size,
                                    pic_size, computer)
train_loader = loader.dataloaders['train']
test_loader = loader.dataloaders['val']
train_size = loader.dataset_sizes['train']
test_size = loader.dataset_sizes['val']

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
        lr = lr / 10
    elif epoch > 150:
        lr = lr / 10
    elif epoch > 120:
        lr = lr / 10
    elif epoch > 90:
        lr = lr / 10
    elif epoch > 60:
        lr = lr / 10
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


#def save_models(epoch,test_corrects):
#    torch.save(model.state_dict(), "cifar10model" + test_corrects +"_{}.model".format(epoch))
#    print("Checkpoint saved")

def save_models(epoch, loss, save_point):
    print("save path types = ",str(type(dataPathRoot))+"\t",str(type(epoch))+"\t",str(type(save_point)))
    save_PATH = dataPathRoot + "/saved_models/" + "Birdies_model_{}_".format(epoch) + "_best_" \
                                + str(save_point) + "_loss_" + str(loss.detach().cpu().numpy()) + ".model"
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }
    torch.save(checkpoint, save_PATH)
    print("Checkpoint saved")
    if (os.path.exists(save_PATH)):
        print("verified save ", save_PATH)


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
    test_acc = test_acc_abs.cpu().numpy() / test_size
    return (test_acc, test_acc_abs)


def train(num_epochs):
    global best_acc, train_acc, train_loss
    best_acc = 0

    since = time.time()
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
            #print("outputs ", outputs.shape," labels ",labels.shape)
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
        train_acc = train_acc.cpu().numpy() / train_size
        train_loss = train_loss / train_size

        # Evaluate on the test set
        results = test()
        test_acc = results[0]
        test_acc_abs = results[1]

            # Save the model if the test acc is greater than our current best
        if test_acc_abs > best_acc and epoch > 1:
                save_models(epoch,loss,str(test_acc_abs.cpu().numpy()))
                best_acc = test_acc_abs

            # Print the metrics
        time_elapsed = time.time() - since
        print("Epoch {}, Train Accuracy: {:.1%} , TrainLoss: {:.4f} , Test Accuracy: {:.1%},"
              "Test Corrects: {}".format(epoch, train_acc, train_loss, test_acc, test_acc_abs),
              ' time {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600,(time_elapsed // 60) % 60, time_elapsed % 60))


        View_Test.test(model,deploy_test, validate_path)


if __name__ == "__main__":

    # ------------------------------------------------------------------
    #  fixed prediction == labels.data,
    #-------------------------------------------------------------------
    load_latest_saved_model("new")
    train(200)
