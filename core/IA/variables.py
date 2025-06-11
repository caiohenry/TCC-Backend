# Imports
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch.nn as nn

# Hiper parameters
BATCH_SIZE = 100
NUMBER_EPOCHS = 20
CRITERION = nn.CrossEntropyLoss()

# Train dataset path
path_dataset_train = "/home/caio/Área de Trabalho/TCC/tcc_cnn_ia_backend/core/IA/datasheet/train"

# Test dataset path
path_dataset_test = "/home/caio/Área de Trabalho/TCC/tcc_cnn_ia_backend/core/IA/datasheet/test"


# Image transform - Normalizing so that the images have the same dimension standard (255 X 255)
image_transform = T.Compose([
    T.Resize((255, 255)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Set image preprocessing parameters and load images from the folder - Train
dataset_train = ImageFolder(
    path_dataset_train, 
    transform = image_transform
)

# Set image preprocessing parameters and load images from the folder - Test
dataset_test = ImageFolder(
    path_dataset_test, 
    transform = image_transform
)


