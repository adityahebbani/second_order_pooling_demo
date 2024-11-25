import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, classification_report
from itertools import chain
from layers import SecondOrderPooling
import numpy as np 
import torch.nn as nn
from fashion_train import FashionCNN

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset and DataLoader
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=
                                              transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

model = FashionCNN()
model.load_state_dict(torch.load("fashion_cnn.pth"))
model.to(device)
model.eval()
print("Model loaded.")

# Testing and evaluation
predictions_list = []
labels_list = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        test = Variable(images)
        outputs = model(test)
        predictions = torch.max(outputs, 1)[1]
        predictions_list.append(predictions)
        labels_list.append(labels)

# Flatten predictions and labels
predictions_l = list(chain.from_iterable([p.tolist() for p in predictions_list]))
labels_l = list(chain.from_iterable([l.tolist() for l in labels_list]))

# Confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(labels_l, predictions_l))
print("\nClassification Report:")
print(classification_report(labels_l, predictions_l))
