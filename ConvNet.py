from torch import nn
import UnitNet

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
        #self.unit3 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size,
        #                          out_channels=self.pic_size)
        self.pool1 = nn.MaxPool2d(kernel_size=self.pooling_factor)

        self.unit4 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size,
                                  out_channels=self.pic_size * 2)
        self.unit5 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 2,
                                  out_channels=self.pic_size * 2)
        self.unit6 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 2,
                                  out_channels=self.pic_size * 2)
        #self.unit7 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 2,
        #                         out_channels=self.pic_size * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=self.pooling_factor)

        self.unit8 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 2,
                                  out_channels=self.pic_size * 4)
        self.unit9 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
                                  out_channels=self.pic_size * 4)
        self.unit10 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
                                   out_channels=self.pic_size * 4)
        #self.unit11 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
        #                           out_channels=self.pic_size * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=self.pooling_factor)

        #self.unit12 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
         #                          out_channels=self.pic_size * 4)
        #self.unit13 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
        #                           out_channels=self.pic_size * 4)
        #self.unit14 = UnitNet.Unit(self.UnitArgs, in_channels=self.pic_size * 4,
        #                           out_channels=self.pic_size * 4)
        self.avgpool = nn.AvgPool2d(kernel_size=(self.pooling_factor * 2) + 1)

        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3,
                                 # self.pool1,
                                 self.unit4, self.unit5, self.unit6,
                                 self.unit7,
                                 # self.pool2,
                                 self.unit8, self.unit9,
                                 self.unit10)
                                 #, self.unit11, self.pool3,
                                 #self.unit12, self.unit13, self.unit14,
                                 #self.avgpool)

        #  self.fc = nn.Linear(in_features=480, out_features=self.num_classes)
        self.fc = nn.Linear(in_features=int(self.pic_size * 4), out_features=self.num_classes)
        #self.fc = nn.Linear(in_features=int(self.pic_size / (self.pooling_factor ** 3 * \
        #                                                (self.pooling_factor * 2 + 1)) * \
        #                                    self.pic_size / (self.pooling_factor ** 3 * \
        #                                                (self.pooling_factor * 2 + 1)) * \
        #                                    self.pic_size * 4), out_features=self.num_classes)

    def forward(self, input):
        output = self.net(input)
        #  output = output.view(-1,480)
        output = output.view(-1,int(self.pic_size * 4))
        #output = output.view(-1,
        #                     int(self.pic_size / (self.pooling_factor ** 3 * \
        #                                     (self.pooling_factor * 2 + 1)) * \
        #                         self.pic_size / (self.pooling_factor ** 3 * \
        #                                     (self.pooling_factor * 2 + 1)) * \
        #                         self.pic_size * 4))
        output = self.fc(output)
        return output
