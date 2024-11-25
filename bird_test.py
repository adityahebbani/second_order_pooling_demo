import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from layers import SecondOrderPooling  # Ensure you have this module
from bird_train import CUB200CNN  # Import your model class

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset path
test_dir = "./CUB_200_2011/test"

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match the input size used during training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load test dataset
test_dataset = ImageFolder(root=test_dir, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# Initialize model and load weights
model = CUB200CNN().to(device)
model.load_state_dict(torch.load("cub200_cnn.pth"))
model.eval()

# Class mapping
class_names = test_dataset.classes

# Initialize variables for tracking performance
all_labels = []
all_predictions = []
total = 0
correct = 0

print("Starting testing...")

# Testing loop
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f"Processing batch {batch_idx + 1}/{len(test_loader)}...")
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Update metrics
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store predictions and true labels for analysis
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Calculate overall accuracy
accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")

# Generate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

# Generate classification report
class_report = classification_report(all_labels, all_predictions, target_names=class_names)
print("\nClassification Report:\n")
print(class_report)

# Display a few test samples with predictions
print("Displaying sample predictions...")
num_samples = min(5, len(test_dataset))
for i in range(num_samples):
    image = test_dataset[i][0]  # Access the image tensor
    label = test_dataset[i][1]  # Access the ground-truth label
    
    # Transform the image back to displayable format
    image_np = image.permute(1, 2, 0).numpy()  # Convert CHW to HWC
    image_np = (image_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # Denormalize
    image_np = np.clip(image_np, 0, 1)  # Clip values to range [0, 1]
    
    # Predict
    model_input = image.unsqueeze(0).to(device)  # Add batch dimension
    output = model(model_input)
    _, predicted_label = torch.max(output, 1)
    
    # Plot the image with prediction and true label
    plt.imshow(image_np)
    plt.title(f"True: {class_names[label]}, Predicted: {class_names[predicted_label.item()]}")
    plt.axis("off")
    plt.show()
