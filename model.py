import torch.nn as nn
from collections import OrderedDict


class SCNnet(nn.Module):
    
    def __init__(self):
        super(SCNnet, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(3, 3), padding = 1)),
            ('bn1',nn.BatchNorm2d(6)),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            ('c3', nn.Conv2d(6, 16, kernel_size=(3, 3), padding = 1)),
            ('bn2',nn.BatchNorm2d(16)),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('bn3',nn.BatchNorm2d(120)),
            ('relu5', nn.ReLU()),
            
            ('deconv',nn.ConvTranspose2d(120, 512, kernel_size=3, stride=1, padding=0, output_padding=0, groups=1, bias=False)),
            ('bn4',nn.BatchNorm2d(512)),
            ('relu7', nn.ReLU()),
            ('s5', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            
            ('c7', nn.Conv2d(512, 512, kernel_size=(3, 3))),
            ('bn5',nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU())
            
        ]))


        self.fc = nn.Sequential(OrderedDict([
            ('f6_1', nn.Linear(512, 84)),
            ('relu11', nn.ReLU()),
            ('f7_1', nn.Linear(84,15)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output