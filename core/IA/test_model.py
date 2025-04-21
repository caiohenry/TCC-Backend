# Imports
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# First-party imports
from model_cnn import CNN
from variables import *


# Variables of test model
correct = 0
total = 0
true_labels = []
pred_labels = []

# Loading the Test Loader for testing
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
 
# CNN Model Instance	
cnn = CNN()

# Loading model
cnn.load_state_dict(torch.load('/home/caio/Área de Trabalho/TCC-2/tcc_cnn_ia_backend/core/IA/models/acuracia_97_model_3.pth'))

# Putting model into evaluation mode
cnn.eval()

# Dataset class
class_names = dataset_test.classes


# Context manager that disables gradient calculation
with torch.no_grad():

    # For earch image and label in test_loader
    for images, labels in test_loader:

        # Obtain outputs
        outputs = cnn(images)

        # Applying soft max
        _, predicted = torch.max(outputs.data, 1)

        # Save true and pred labels in list to generate confusion matrix
        true_labels.extend(labels.tolist())
        pred_labels.extend(predicted.tolist())

        # Save the total images test and correct images test
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Display with accuracy
print(f"\nAcurácia no conjunto de teste: {100 * correct / total:.2f}%")

# Generate confusion matrix
confusion_matrix_test = confusion_matrix(true_labels, pred_labels)

# Display the matrix with class names
disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_test, display_labels = class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.subplots_adjust(left=0, right=0.95, top=0.9, bottom=0.3)
plt.xticks(rotation=45)

# Save image
plt.savefig("matriz_confusao.png")