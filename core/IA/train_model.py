# Imports
import torch

# First-party imports
from model_cnn import CNN
from variables import *


# Loading the Train Loader for training
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
 
# CNN Model Instance	
cnn = CNN()

# Defining the optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)


# Training Phase
for epoch in range(NUMBER_EPOCHS):

    # Train model
    model = cnn.train()

    # For batch in train_loader
    for batch_idx, (X_branch, Y_branch) in enumerate(train_loader):

        # Call optimizer
        optimizer.zero_grad()

        # Obtain outputs from model
        outputs = cnn(X_branch)
        
        # Applying the loss function
        loss = CRITERION(outputs, Y_branch)
        loss.backward()

        # Applying step train
        optimizer.step()


# Save model before train
torch.save(model.state_dict(), f'/home/caio/Área de Trabalho/TCC/tcc_cnn_ia_backend/core/IA/models/model_3.pth')
