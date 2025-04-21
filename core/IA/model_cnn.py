# Imports
import torch.nn as nn
import torch.nn.functional as F
import torch


# CNN Model Class
class CNN(nn.Module):

    # Init Method
    def __init__(self):
        super(CNN, self).__init__()

        # First convolution layer
        self.conv1 = nn.Sequential(

            # Receives images with 3 channels (RGB)
            # Produces 16 feature maps
            # Uses 5x5 kernels
            # Stride 1 (convolution stride)
            # Padding 2 (to maintain spatial dimensions)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),

            # Batch normalization for 16 channels
            nn.BatchNorm2d(16),

            # Activation function
            nn.ReLU(),

            # Max pooling with 2x2 window and stride 2 (reduces dimension by half)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second convolution layer
        self.conv2 = nn.Sequential(

            # Receives images with 16 channels (RGB)
            # Produces 32 feature maps
            # Uses 5x5 kernels
            # Stride 1 (convolution stride)
            # Padding 2 (to maintain spatial dimensions)
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),

            # Batch normalization for 32 channels
            nn.BatchNorm2d(32),

            # Activation function
            nn.ReLU(),

            # Max pooling with 2x2 window and stride 2 (reduces dimension by half)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        # Third convolution layer
        self.conv3 = nn.Sequential(

            # Receives images with 32 channels (RGB)
            # Produces 64 feature maps
            # Uses 3x3 kernels
            # Stride 1 (convolution stride)
            # Padding 1 (to maintain spatial dimensions)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),

            # Batch normalization for 64 channels
            nn.BatchNorm2d(64),

            # Activation function
            nn.ReLU(),

            # Max pooling with 2x2 window and stride 2 (reduces dimension by half)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Combine all convolutional layers into a single sequence
        self.convs = nn.Sequential(self.conv1, self.conv2, self.conv3)

        # Call method to calculate output size
        x = self.convs(torch.randn(1, 3, 255, 255))

        # Variable to store size of flattened output
        self._to_linear = x.view(1, -1).shape[1]

        # First fully connected layer
        self.fc1 = nn.Linear(self._to_linear, 256)

        # Dropout layer with 50% chance of zeroing a neuron (avoids overfitting)
        self.dropout = nn.Dropout(0.5)

        # Second fully connected layer
        self.fc2 = nn.Linear(256, 2)

    # Network forward flow
    def forward(self, x):

        # Passing the input through convolutional layers
        x = self.convs(x)

        # Flatten output to 1D vector
        x = x.view(x.size(0), -1)

        # Goes through the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply dropout
        x = self.dropout(x)

        # Passing through the final layer
        x = self.fc2(x)

        # Return output
        return x
