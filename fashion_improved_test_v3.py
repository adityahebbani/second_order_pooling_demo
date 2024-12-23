import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import RevisedCNN_256_V3  # Updated model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm

# Define the GradCAM class
class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initializes the GradCAM object.

        Args:
            model (torch.nn.Module): The trained model.
            target_layer (torch.nn.Module): The layer to target for Grad-CAM.
        """
        self.model = model
        self.gradient = None
        self.activation = None
        self.target_layer = target_layer
        self.hook_layers()

    def hook_layers(self):
        """
        Registers forward and backward hooks to capture activations and gradients.
        """
        def forward_hook(module, input, output):
            self.activation = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradient = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        """
        Generates the Grad-CAM heatmap for a given input and target class.

        Args:
            input_image (torch.Tensor): The input image tensor.
            target_class (int): The target class index.

        Returns:
            np.ndarray: The generated heatmap.
        """
        self.model.zero_grad()
        output = self.model(input_image)
        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradient.cpu().numpy()[0]  # [C, H, W]
        activations = self.activation.cpu().numpy()[0]  # [C, H, W]

        weights = np.mean(gradients, axis=(1, 2))  # [C]

        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam

        return cam

def load_model(model_path, device):
    """
    Loads the trained model from the specified path.

    Args:
        model_path (str): Path to the saved model checkpoint.
        device (torch.device): The device to load the model on.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = RevisedCNN_256_V3()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_tensor):
    """
    Preprocesses the image tensor for visualization.

    Args:
        image_tensor (torch.Tensor): The image tensor.

    Returns:
        np.ndarray: The preprocessed image in BGR format.
    """
    image = image_tensor.cpu().numpy()[0].transpose(1, 2, 0)  # [H, W, C]
    image = (image * 0.5) + 0.5  # Unnormalize to [0,1]
    image = np.clip(image, 0, 1)
    image = np.uint8(255 * image)  # [H, W, C] in [0, 255]
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel BGR
    return image

def display_gradcam(image, heatmap, label, prediction, idx, class_names):
    """
    Displays the image with the Grad-CAM heatmap overlaid.

    Args:
        image (np.ndarray): The original image.
        heatmap (np.ndarray): The Grad-CAM heatmap.
        label (str): The true class label.
        prediction (str): The predicted class label.
        idx (int): The index of the example.
        class_names (list): List of class names.
    """
    superimposed_img = cv2.addWeighted(heatmap, 0.4, image, 0.6, 0)

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Image {idx} - True: {label}, Pred: {prediction}')
    plt.axis('off')
    plt.show()

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load the trained model
    model_path = './checkpoints/revised_cnn_256_v3_best.pth'  # Update path if necessary
    try:
        model = load_model(model_path, device)
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Define the transform (same as training)
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the test dataset
    test_dir = './data/FashionMNIST_256_split/test'  # Update path if necessary
    if not os.path.exists(test_dir):
        print(f"Test directory not found at {test_dir}")
        return

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f'Number of test samples: {len(test_loader.dataset)}')

    # Initialize GradCAM
    # Adjust target_layer based on your model architecture
    # Example: If your RevisedCNN_256_V3 has layers like conv1, conv2, conv3
    try:
        target_layer = model.conv3  # Last convolutional layer
    except AttributeError:
        print("The specified target_layer does not exist in the model. Please adjust the target_layer.")
        return

    grad_cam = GradCAM(model, target_layer)

    # Perform inference and collect predictions
    all_preds = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Inference", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Generate Grad-CAM examples
    # Select a few correct and incorrect predictions to visualize
    num_examples = 5
    correct_examples = []
    incorrect_examples = []
    for idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for i in range(len(inputs)):
            if preds[i] == labels[i] and len(correct_examples) < num_examples:
                correct_examples.append((inputs[i:i+1], labels[i].item(), preds[i].item()))
            elif preds[i] != labels[i] and len(incorrect_examples) < num_examples:
                incorrect_examples.append((inputs[i:i+1], labels[i].item(), preds[i].item()))
            if len(correct_examples) >= num_examples and len(incorrect_examples) >= num_examples:
                break
        if len(correct_examples) >= num_examples and len(incorrect_examples) >= num_examples:
            break

    # Function to map class indices to class names
    class_names = test_dataset.classes

    # Visualize correct predictions
    print("\nGrad-CAM for Correct Predictions:")
    for idx, (input_img, true_label, pred_label) in enumerate(correct_examples, 1):
        cam = grad_cam.generate_cam(input_img, pred_label)
        image = preprocess_image(input_img)

        heatmap = np.uint8(255 * cam)  # Convert to [0,255]
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        display_gradcam(image, heatmap, class_names[true_label], class_names[pred_label], idx, class_names)

    # Visualize incorrect predictions
    print("\nGrad-CAM for Incorrect Predictions:")
    for idx, (input_img, true_label, pred_label) in enumerate(incorrect_examples, 1):
        cam = grad_cam.generate_cam(input_img, pred_label)
        image = preprocess_image(input_img)

        heatmap = np.uint8(255 * cam)  # Convert to [0,255]
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        display_gradcam(image, heatmap, class_names[true_label], class_names[pred_label], idx, class_names)

if __name__ == "__main__":
    main()