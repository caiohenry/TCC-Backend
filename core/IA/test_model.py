# Imports
import torch
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score,
    precision_score, roc_curve, auc
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# First-party imports
from model_cnn import CNN
from variables import *

# Variables
correct = 0
total = 0
true_labels = []
pred_labels = []
pred_scores = []
class_labels_pt = ['Baixa Probabilidade de Tumor', 'Alta Probabilidade de Tumor']
pred_labels_pt = ['Baixa Probabilidade\n de Tumor', 'Alta Probabilidade\n de Tumor']

# Test loader
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

# Load model
cnn = CNN()
cnn.load_state_dict(torch.load('/home/caio/Área de Trabalho/TCC/tcc_cnn_ia_backend/core/IA/models/acuracia_97_model_3.pth'))
cnn.eval()

class_names = dataset_test.classes

# Disable gradient
with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)

        softmax_scores = torch.nn.functional.softmax(outputs, dim=1)
        pred_scores.extend(softmax_scores[:, 1].cpu().numpy())

        true_labels.extend(labels.tolist())
        pred_labels.extend(predicted.tolist())

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Accuracy
print(f"\nAcurácia no conjunto de teste: {100 * correct / total:.2f}%")


f1 = f1_score(true_labels, pred_labels, average='weighted')
recall = recall_score(true_labels, pred_labels, average='weighted')
precision = precision_score(true_labels, pred_labels, average='weighted')

print(f"F1 Score: {f1:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")


conf_matrix = confusion_matrix(true_labels, pred_labels)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Plot matrix
df_cm = pd.DataFrame(conf_matrix_percent, index=pred_labels_pt, columns=class_labels_pt)
fmt_values = np.vectorize(lambda x: f"{x:.2f}%")(conf_matrix_percent)
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=fmt_values, fmt='', cmap='Blues', cbar=False)

plt.title("Matriz de Confusão (%)")
plt.ylabel("Classe Verdadeira") 
plt.xlabel("Classe Prevista")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("matriz_confusao_percentual.png")
plt.show()

true_bin = np.array(true_labels)
if set(true_bin) == {1, 2}:
    true_bin = true_bin - 1


fpr, tpr, _ = roc_curve(true_bin, pred_scores)
roc_auc = auc(fpr, tpr)

print(f"AUC: {roc_auc:.2f}")
# Plot ROC
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('TFP')
plt.ylabel('TVP')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid()
plt.savefig("curva_roc.png")
