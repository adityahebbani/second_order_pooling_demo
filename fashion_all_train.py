import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import CNN_28, CNN_56, CNN_128, CNN_256
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Resolutions and models
resolutions = {
    28: CNN_28(),
    56: CNN_56(),
    128: CNN_128(),
    256: CNN_256()
}

# Hyperparameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001

for size, model in resolutions.items():
    print(f'Training model for resolution {size}x{size}')

    # Define transform without resizing
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if size == 28:
        # Load FashionMNIST dataset as usual
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    else:
        # Load custom dataset from data directory
        data_dir = f'./data/FashionMNIST_{size}_split/train'
        train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Move model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), f'model_{size}x{size}.pth')