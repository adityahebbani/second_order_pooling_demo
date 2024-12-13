# nih_xray_test_pytorch_single_label.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from layers import SecondOrderPooling  # Ensure you have this module if used in the model
from xray_train_pytorch import NIHChestXrayCNN  # Make sure this matches your training script's model definition
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Dataset path
DATA_DIR = r"C:\Users\Total\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3"
CSV_PATH = os.path.join(DATA_DIR, "Filtered_Data_Entry.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "filtered_images")

# Define single-label classes (15 classes including "No Finding")
CLASSES = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

class NIHChestXraySingleLabelDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file with annotations.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_path)
        # Filter to single-label entries if necessary
        self.data = self.data[self.data['Finding Labels'].apply(lambda x: len(x.split('|')) == 1)].reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.classes = CLASSES
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self._prepare_labels()

    def _prepare_labels(self):
        """Convert labels to single integer encoding."""
        self.data['label'] = self.data['Finding Labels'].apply(self._encode_label)

    def _encode_label(self, label_str):
        """Encode a single label into class index."""
        label = label_str.strip()
        if label in self.class_to_idx:
            return self.class_to_idx[label]
        else:
            # Assign to "No Finding" if label not recognized
            return self.class_to_idx["No Finding"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx]['Image Index'])
        try:
            image = Image.open(img_name).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a black image in case of error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        label = self.data.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# Define denormalization function for visualization
def denormalize(img, mean, std):
    """
    Denormalize an image tensor.
    Args:
        img (torch.Tensor): Normalized image tensor.
        mean (tuple): Mean for each channel.
        std (tuple): Standard deviation for each channel.
    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

# Define Dice Score function
def dice_score(y_true, y_pred, num_classes, smooth=1e-6):
    """
    Compute Dice Score for multi-class classification.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        num_classes (int): Number of classes.
        smooth (float): Smoothing factor to avoid division by zero.
        
    Returns:
        float: Mean Dice Score over all classes.
    """
    dice = 0.0
    for cls in range(num_classes):
        intersection = np.sum((y_true == cls) & (y_pred == cls))
        denominator = np.sum(y_true == cls) + np.sum(y_pred == cls)
        dice_cls = (2. * intersection + smooth) / (denominator + smooth)
        dice += dice_cls
    return dice / num_classes

# Define function to plot sample predictions
def plot_sample_predictions(model, dataset, device, class_names, num_samples=15):
    """
    Plot a grid of sample images with their true and predicted labels.
    
    Args:
        model (nn.Module): Trained PyTorch model.
        dataset (Dataset): Dataset to sample images from.
        device (torch.device): Device to perform computations on.
        class_names (list): List of class names.
        num_samples (int): Number of sample images to display.
    """
    model.eval()
    samples_plotted = 0
    cols = 5
    rows = int(np.ceil(num_samples / cols))
    plt.figure(figsize=(cols * 4, rows * 4))
    
    with torch.no_grad():
        for i in range(len(dataset)):
            if samples_plotted >= num_samples:
                break
            image, label = dataset[i]
            image_input = image.unsqueeze(0).to(device)  # Add batch dimension
            output = model(image_input)
            _, pred = torch.max(output, 1)
            pred_label = pred.item()
            true_label = label
            
            # Denormalize image for visualization
            image_denorm = denormalize(image.cpu().clone(), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            image_denorm = image_denorm.numpy()
            image_denorm = np.transpose(image_denorm, (1, 2, 0))  # CHW to HWC
            image_denorm = np.clip(image_denorm, 0, 1)  # Clip to [0,1] range
            
            # Plot the image with prediction and true labels
            ax = plt.subplot(rows, cols, samples_plotted + 1)
            plt.imshow(image_denorm)
            plt.axis("off")
            title_color = 'green' if pred_label == true_label else 'red'
            plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", color=title_color, fontsize=10)
            samples_plotted += 1
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # We will test all six models, similar to how we trained them
    # pool_modes = ['second_order', 'max']
    # image_sizes = [256, 512, 1024]
    pool_modes = ["global_average"]
    image_sizes = [256]

    for pool_mode in pool_modes:
        for img_size in image_sizes:
            print(f"\n=== Testing model with {pool_mode} pooling at resolution {img_size}x{img_size} ===")

            # Set up transforms according to image size used in training
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            # Load test dataset
            test_dataset = NIHChestXraySingleLabelDataset(csv_path=CSV_PATH, image_dir=IMAGE_DIR, transform=transform)
            test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)

            # Model path
            model_path = f"nih_xray_cnn_best_{pool_mode}_{img_size}.pth"
            if not os.path.exists(model_path):
                print(f"Model file not found at {model_path}. Skipping...")
                continue

            # Load the model with the correct configuration
            model = NIHChestXrayCNN(num_classes=len(CLASSES), pool_mode=pool_mode).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print("Model loaded successfully.")

            print("Starting testing...")
            all_labels = []
            all_predictions = []

            # Testing loop
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(preds.cpu().numpy())

            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)

            # Compute metrics
            f1 = f1_score(all_labels, all_predictions, average='macro')
            dice = dice_score(all_labels, all_predictions, num_classes=len(CLASSES))
            accuracy = accuracy_score(all_labels, all_predictions)
            class_report = classification_report(all_labels, all_predictions, target_names=CLASSES, zero_division=0, output_dict=True)
            cm = confusion_matrix(all_labels, all_predictions)

            # Print Metrics
            print("\n=== Evaluation Metrics ===")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score (Macro): {f1:.4f}")
            print(f"Dice Score (Macro): {dice:.4f}\n")

            # Convert classification report to DataFrame for better formatting
            class_report_df = pd.DataFrame(class_report).transpose()
            print("Classification Report:")
            print(class_report_df)

            # Plot Confusion Matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=CLASSES, yticklabels=CLASSES)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix for {pool_mode} Pooling at {img_size}x{img_size}')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

            # Save metrics to a file
            metrics_filename = f"nih_xray_test_metrics_{pool_mode}_{img_size}.txt"
            with open(metrics_filename, "w") as f:
                f.write(f"=== Evaluation Metrics ===\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"F1 Score (Macro): {f1:.4f}\n")
                f.write(f"Dice Score (Macro): {dice:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(class_report_df.to_string())
                f.write("\n\nConfusion Matrix:\n")
                f.write(np.array2string(cm))
                f.write("\n")

            # Save Confusion Matrix plot
            cm_filename = f"confusion_matrix_{pool_mode}_{img_size}.png"
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=CLASSES, yticklabels=CLASSES)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title(f'Confusion Matrix for {pool_mode} Pooling at {img_size}x{img_size}')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(cm_filename)
            plt.close()

            # Visualization of sample predictions
            print("Starting visualization of sample predictions...")
            plot_sample_predictions(model, test_dataset, device, CLASSES, num_samples=15)
            print(f"Finished testing and visualization for {pool_mode} pooling at {img_size}x{img_size} resolution.")
