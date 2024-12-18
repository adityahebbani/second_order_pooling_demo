import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model import CNN_28, CNN_56, CNN_128, CNN_256
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Resolutions and models
resolutions = {
    28: CNN_28(),
    56: CNN_56(),
    128: CNN_128(),
    256: CNN_256()
}

batch_size = 64

for size, model in resolutions.items():
    print(f'Testing model for resolution {size}x{size}')

    # Transform for the current resolution
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if size == 28:
        # Load FashionMNIST dataset as usual
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        # Load custom dataset from data directory
        data_dir = f'./data/FashionMNIST_{size}_split/test'
        test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Load the saved model
    model.load_state_dict(torch.load(f'model_{size}x{size}.pth'))
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Classification report
    print(f'Classification Report for resolution {size}x{size}:')
    print(classification_report(all_labels, all_preds))

    # Confusion matrix
    print(f'Confusion Matrix for resolution {size}x{size}:')
    print(confusion_matrix(all_labels, all_preds))

    # Display 6 images with their predicted and true labels
    examples = 0
    fig, axes = plt.subplots(1, 6, figsize=(15, 3))
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for idx in range(images.size(0)):
                if examples >= 6:
                    break
                img = images[idx].cpu().numpy().squeeze()
                true_label = labels[idx].item()
                pred_label = predicted[idx].item()
                axes[examples].imshow(img, cmap='gray')
                axes[examples].set_title(f'True: {true_label}\nPred: {pred_label}')
                axes[examples].axis('off')
                examples += 1
            if examples >= 6:
                break
    plt.suptitle(f'Resolution {size}x{size}')
    plt.show()