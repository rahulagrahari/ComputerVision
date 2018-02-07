from torchvision import models
import torch.nn as nn
import pdb


class Demo_Model(nn.Module):
    def __init__(self, nClasses=200):
        super(Demo_Model, self).__init__();

        self.conv_1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        #64x64x32
        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(32);
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32X32X32
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        # 32X32X32
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(32);
        '''self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2)

        self.conv_3 = nn.Conv2d(32,32,kernel_size=5,stride=1, padding = 2)
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(32);'''

        self.fc_1 = nn.Linear(32768, 1024);
        # self.fc_1 = nn.Linear(8192, 200);
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p=0.5);
        self.fc_2 = nn.Linear(1024, nClasses);

    def forward(self, x):
        # pdb.set_trace();
        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)
        y = self.pool_1(y)

        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        '''y = self.pool_2(y)
        
        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)'''

        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return (y)

class MyModel(nn.Module):
    def __init__(self, nClasses=200):
        super(MyModel, self).__init__();
        image_size = 64
        filter_size = 0
        strid = 0
        padding = 2
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=4)
        self.relu_1 = nn.ReLU(True);
        self.batch_norm_1 = nn.BatchNorm2d(16);
        output = ((64 - 3 + (2 * 4)) / 1) + 1
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1
        print('+++++++', output, '++++++++')

        self.conv_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=4)
        output = ((output - 3 + (2 * 4)) / 1) + 1
        self.relu_2 = nn.ReLU(True);
        self.batch_norm_2 = nn.BatchNorm2d(32);
        self.pool_2 = nn.MaxPool2d(kernel_size = 2, stride =2)
        output = ((output - 2 + (2 * 0)) / 2) + 1
        print('+++++++', output, '++++++++')

        self.conv_3 = nn.Conv2d(32, 64,kernel_size=3,stride=1, padding=4)
        output = ((output - 3 + (2 * 4)) / 1) + 1
        self.relu_3 = nn.ReLU(True);
        self.batch_norm_3 = nn.BatchNorm2d(64);
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1
        print('+++++++', output, '++++++++')

        self.conv_4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=4)
        output = ((output - 3 + (2 * 4)) / 1) + 1
        self.relu_4 = nn.ReLU(True);
        self.batch_norm_4 = nn.BatchNorm2d(128);
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        output = ((output - 2 + (2 * 0)) / 2) + 1

        self.conv_5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4)
        output = ((output - 3 + (2 * 4)) / 1) + 1
        self.relu_5 = nn.ReLU(True);
        self.batch_norm_5 = nn.BatchNorm2d(128);
        # self.pool5 = nn.MaxPool2d()
        print('+++++++',output,'++++++++')
        print(output*output*128)

        fc_input = output*output*128
        self.fc_1 = nn.Linear(fc_input, 1024);
        # self.fc_1 = nn.Linear(8192, 200);
        self.relu_6 = nn.Softmax()
        self.batch_norm_6 = nn.BatchNorm1d(1024);
        self.dropout_1 = nn.Dropout(p=0.5);
        self.fc_2 = nn.Linear(1024, nClasses);

    def forward(self, x):
        # pdb.set_trace();
        y = self.conv_1(x)
        y = self.relu_1(y)
        y = self.batch_norm_1(y)
        y = self.pool_1(y)

        y = self.conv_2(y)
        y = self.relu_2(y)
        y = self.batch_norm_2(y)
        y = self.pool_2(y)

        y = self.conv_3(y)
        y = self.relu_3(y)
        y = self.batch_norm_3(y)
        y = self.pool_3(y)

        y = self.conv_4(y)
        y = self.relu_4(y)
        y = self.batch_norm_4(y)
        y = self.pool_4(y)

        y = self.conv_5(y)
        y = self.relu_5(y)
        y = self.batch_norm_5(y)

        # print('-------------', y.size(0))
        y = y.view(y.size(0), -1)
        y = self.fc_1(y)
        y = self.relu_6(y)
        y = self.batch_norm_6(y)
        y = self.dropout_1(y)
        y = self.fc_2(y)
        return (y)


def resnet18(pretrained=True):
    return models.resnet18(pretrained)


def demo_model():
    return Demo_Model();
