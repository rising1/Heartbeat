from torch import nn


class Unit(nn.Module):
    def __init__(self, argslist, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = argslist[0]
        self.stride_pixels = argslist[1]
        self.padding_pixels = argslist[2]
        self.dropout_factor = argslist[3]

        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=
                              self.out_channels, kernel_size=self.kernel_sizes,
                              stride=self.stride_pixels, padding=self.padding_pixels)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(p=self.dropout_factor)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        output = self.do(output)

        return output