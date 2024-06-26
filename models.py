import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models

class SModel(nn.Module):
    def __init__(self):
        super(SModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=7, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(4)

        self.maxpool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(8)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(8 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        with torch.profiler.record_function("CONV_BLOCK_1"):
            # CONV_BLOCK_1
            out = self.conv1(x)
            out = self.batchnorm1(out)
            out = self.relu(out)
            out = self.maxpool1(out)

        with torch.profiler.record_function("CONV_BLOCK_2"):
            # CONV_BLOCK_2
            out = self.conv2(out)
            out = self.batchnorm2(out)
            out = self.relu(out)
            out = self.maxpool2(out)

        with torch.profiler.record_function("MLP_LAYER"):
            # MLP_LAYER
            out = self.flatten(out)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)

        return out

class MModel(nn.Module):
    def __init__(self):
        super(MModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.maxpool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32) 
        self.fc4 = nn.Linear(32, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        with torch.profiler.record_function("CONV_BLOCK_1"):
            # CONV_BLOCK_1
            out = self.conv1(x)
            out = self.batchnorm1(out)
            out = F.relu(out)
            out = self.maxpool1(out)

        with torch.profiler.record_function("CONV_BLOCK_2"):
            # CONV_BLOCK_2
            out = self.conv2(out)
            out = self.batchnorm2(out)
            out = F.relu(out)
            out = self.maxpool2(out)

        with torch.profiler.record_function("MLP_LAYER"):
            # MLP_LAYER
            out = self.flatten(out)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            out = self.relu(out)
            out = self.fc4(out)

        return out

class LModel(nn.Module):
    def __init__(self):
        super(LModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.maxpool1  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * 9 * 9, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):

        with torch.profiler.record_function("CONV_BLOCK_1"):
            # CONV_BLOCK_1
            out = self.conv1(x)
            out = self.batchnorm1(out)
            out = F.relu(out)
            out = self.maxpool1(out)

        with torch.profiler.record_function("CONV_BLOCK_2"):
            # CONV_BLOCK_2
            out = self.conv2(out)
            out = self.batchnorm2(out)
            out = F.relu(out)
            out = self.maxpool2(out)

        with torch.profiler.record_function("MLP_LAYER"):
            # MLP_LAYER
            out = self.flatten(out)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            out = self.relu(out)
            out = self.fc4(out)

        return out


class Model(nn.Module):
    ALLOWED_MODELS = [
        'resnet', 'alexnet', 'googlenet'
    ]

    def __init__(
        self, 
        model_name: str  = 'resnet',
    ) -> None:
        
        super(Model, self).__init__()

        self.model_name = model_name
        if self.model_name == 'resnet':
            self.model = models.resnet34()
        
        elif self.model_name == 'alexnet': 
            self.model = models.alexnet()
        
        elif self.model_name == 'googlenet':
            self.model = models.googlenet()

        else: 
            assert self.model_name not in self.ALLOWED_MODELS, f"{self.model}"

        # All models have 1000 nodes in the output layer because of original ImageNet dataset
        # Need to add more fc layers to make it to 10 for ImageNette dataset

        self.mlp_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(), 
            nn.Linear(512, 64), 
            nn.ReLU(), 
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        # torch.cuda.nvtx.range_push(f'layer:{self.model_name}')
        with torch.profiler.record_function(f"{self.model_name}"):
            outs = self.model(x)


        # torch.cuda.nvtx.range_push(f'layer:mlp_layer')
        with torch.profiler.record_function("MLP_LAYER"):
            outs = self.mlp_layer(outs)

        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_pop()
        return outs
