   # Import needed packages
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# import hawknet_depld


# Hyper-parameters
colour_channels = 3  # used in SimpleNet
no_feature_detectors = 64 # used in Unit
kernel_sizes = 3 # 3 works  # used in Unit
stride_pixels = 1  # used in Unit
padding_pixels = 1  # used in Unit
pooling_factor = 2  # used in SimpleNet
pic_size = 72 # used in SimpleNet
flattener = 16
output_classes = 220  # used in SimpleNet0
learning_rate = 0.00001  # used in HeartbeatCleandecay_cycles = 1  # default to start
weight_decay = 0.0001  # used in HeartbeatClean
dropout_factor = 0.0  # used in Unit
faff = 'false'
# linear_mid_layer = 1024
# linear_mid_layer_2 = 230
num_epochs = 50 # used in HeartbeatClean
snapshot_points = num_epochs / 1
batch_sizes = 32 # used in HeartbeatClean
#  batch_sizes = 6 # used in HeartbeatClean
loadfile = True
print_shape = False

dataPathRoot = './'

# validate_path = 'C:/Users/phfro/PycharmProjects/Heartbeat/bird_list.txt'

computer = "home_laptop"
# deploy_test = hawknet_depld.test_images(12, False)
# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

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
        self.do = nn.Dropout(dropout_factor)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()


    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.do(output)
        output = self.relu(output)
        # output = self.softmax(output)
        return output


class ModelBuilder(nn.Module):
    def __init__(self, SimpleNetArgs):
        super(ModelBuilder, self).__init__()

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
        self.fc = nn.Linear(no_feature_detectors * flattener , output_classes)

    def forward(self, input):
        global print_shape
        output = self.net(input)
        if(print_shape):
            print("net(input) ",output.shape)
        output = output.view(-1, no_feature_detectors * flattener)
        #output = output.view(-1, no_feature_detectors * 4 * 4)
        # output = output.view(-1, no_feature_detectors * 4 )
        # output = output.view(-1, int(no_feature_detectors / 4))
        #print("output.view ",output.shape)
        output = self.fc(output)
        #print("fc_final(output) ",output.shape)
        #output = self.fc2(output)
        #print("fc_final(output) ",output.shape)
        # output = self.fc_final(output)
        if(print_shape):
            print("fc(output) ",output.shape)
        return output


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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

batch_size = batch_sizes


cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
model = ModelBuilder(SimpleNetArgs)

if cuda_avail:
    model.cuda()

loss_fn = nn.CrossEntropyLoss()

def set_print_shape(printit):
    global print_shape
    print_shape = printit
