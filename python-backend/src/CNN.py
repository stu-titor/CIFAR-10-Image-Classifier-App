import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageNeuralNetwork(nn.Module):
    
    def __init__(self, channels = 32, layers = 3, conv_blocks = 4, num_classes = 10, dropout_rate = 0.5):
        super().__init__() 
        
        # Instance variables
        self.convBlocks = nn.ModuleList()
        self.batches = nn.ModuleList()
        self.fcs = nn.ModuleList()

        # First convolutional block and batch normalizations
        self.convBlocks.append(nn.Conv2d(3, channels, 3, padding = 1))
        self.batches.append(nn.BatchNorm2d(channels))

        # Additional convolutional blocks and batch normalizations
        for i in range(conv_blocks - 1):
            self.convBlocks.append(nn.Conv2d(channels, channels * 2, 3, padding = 1))
            self.batches.append(nn.BatchNorm2d(channels * 2))
            channels *= 2
    
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        final_size = 32 // (2 ** conv_blocks)
        flattened_size = channels * final_size * final_size
        self.fcs.append(nn.Linear(flattened_size, channels))
        for i in range(layers - 2):
            self.fcs.append(nn.Linear(channels, channels // 2)) 
            channels //= 2
        self.fcs.append(nn.Linear(channels, num_classes))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Convolutional blocks
        for i in range(len(self.convBlocks)):
            x = self.pool(F.relu(self.batches[i](self.convBlocks[i](x))))
        x = torch.flatten(x, 1)
        
        # Fully connected layers with dropout
        for i in range(len(self.fcs) - 1):
            x = self.dropout(F.relu(self.fcs[i](x)))
        x = self.fcs[-1](x) 
        
        return x