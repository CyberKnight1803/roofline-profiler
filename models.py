import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class SModel(nn.Module):
    def __init__(
        self,
    ): 
        super(SModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=4, 
            kernel_size=7, 
            stride=2, 
            padding=1
        )

        self.maxpool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=4, 
            out_channels=8, 
            kernel_size=5, 
            stride=2
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(8 * 9 * 9, 10)
        self.relu = nn.ReLU()

    def forward(self, x): 
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        out = self.flatten(out)

        out = self.fc1(out)

        return out

class MModel(nn.Module):
    def __init__(self):
        super(MModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=4, 
            kernel_size=7, 
            stride=2, 
            padding=1
        )

        self.maxpool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=4, 
            out_channels=8, 
            kernel_size=5, 
            stride=2
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(8 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

class LModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=4, 
            kernel_size=7, 
            stride=2, 
            padding=1
        )

        self.maxpool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=4, 
            out_channels=8, 
            kernel_size=5, 
            stride=2
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(8 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32) 
        self.fc4 = nn.Linear(32, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.maxpool2(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)

        return out

        
