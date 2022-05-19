from collections import OrderedDict
from torch.nn import Sequential
import torch.nn as nn
import torch
import math
class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.conv1  = nn.Conv2d(channel[1], channel[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1    = nn.BatchNorm2d(channel[0])
        self.relu1  = nn.LeakyReLU()
        
        self.conv2  = nn.Conv2d(channel[0], channel[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(channel[1])
        self.relu2  = nn.LeakyReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out

class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.inchannel = 32
        self.Upsampling16=nn.Upsample(16)
        self.Upsampling32=nn.Upsample(32)
        self.Upsampling64=nn.Upsample(64)
        # 416,416,3 -> 416,416,32
        self.conv1  = nn.Conv2d(3, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(self.inchannel)
        self.relu1  = nn.LeakyReLU()

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], 2)
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128],4)
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], 8)
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], 8)
        # 26,26,512 -> 13,13,1024

        self.conv4_3=Sequential(
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.conv3_2=Sequential(
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.head1=Sequential(
            nn.Conv2d(in_channels=384,out_channels=512,kernel_size=1,stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=5,kernel_size=1,stride=1),
        )

        self.head2=Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1,stride=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1,stride=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=1,stride=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512,out_channels=5,kernel_size=1,stride=1),
        )

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, channel, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(channel[0], channel[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(channel[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), ResBlock(channel)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        
        out2 = self.layer2(x)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out3 = torch.cat([self.Upsampling32(self.conv4_3(out4)),out3],1)

        feature2 = self.head2(out4)

        out2 = torch.cat([self.Upsampling64(self.conv3_2(out3)),out2],1)

        feature1 = self.head1(out2) #大scale预测小物体

        return feature1, feature2
