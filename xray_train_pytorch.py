# nih_xray_train_pytorch_single_label_specific_order.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pytorch_layers import SecondOrderPooling  # Ensure this is the PyTorch implementation
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
DATA_DIR = r"C:\Users\Total\.cache\kagglehub\datasets\nih-chest-xrays\data\versions\3"
CSV_PATH = os.path.join(DATA_DIR, "Filtered_Data_Entry.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "filtered_images")

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
# You can try reducing workers if there's instability. Start with 4, if error persists, reduce to 0 or 1.
NUM_WORKERS = 4
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
PATIENCE = 5

# Define single-label classes
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
class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

def encode_label(label_str):
    """Encode a single label into class index."""
    labels = label_str.split('|')
    label = labels[0].strip()
    if label in class_to_idx:
        return class_to_idx[label]
    else:
        return class_to_idx["No Finding"]

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # Keep only single-label entries
    df_single = df[df['Finding Labels'].apply(lambda x: len(x.split('|')) == 1)].reset_index(drop=True)
    # Encode labels
    df_single['label'] = df_single['Finding Labels'].apply(encode_label)
    return df_single

def compute_class_weights(labels, num_classes):
    """
    Compute class weights inversely proportional to class frequencies.
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    # Avoid division by zero
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1. / class_counts
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.tensor(class_weights, dtype=torch.float32)

class NIHChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.classes = CLASSES
        self.num_classes = len(self.classes)
        self.class_to_idx = class_to_idx

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

class NIHChestXrayCNN(nn.Module):
    def __init__(self, num_classes=15, pool_mode='second_order'):
        super(NIHChestXrayCNN, self).__init__()
        self.pool_mode = pool_mode
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        if self.pool_mode == 'second_order':
            # Second order pooling returns a (C*C) dimension vector
            # Here C=128, so fc input is 128*128=16384
            self.pooling_layer = SecondOrderPooling()
            fc_input_dim = 128 * 128
        elif self.pool_mode == 'global_average':
            # Global average pooling reduces to a single value per channel
            # This gives a vector of dimension 128
            self.pooling_layer = nn.AdaptiveAvgPool2d((1, 1))
            fc_input_dim = 128
        else:
            # Max pooling to a single spatial location
            # This gives a vector of dimension 128
            self.pooling_layer = nn.AdaptiveMaxPool2d((1, 1))
            fc_input_dim = 128

        self.fc1 = nn.Linear(fc_input_dim, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pooling_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(train_loader, val_loader, class_weights, pool_mode, image_size, model_save_path):
    model = NIHChestXrayCNN(num_classes=len(CLASSES), pool_mode=pool_mode).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Training [{pool_mode}, {image_size}x{image_size}]", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Validation [{pool_mode}, {image_size}x{image_size}]", leave=False)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_bar.set_postfix(loss=loss.item())

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = 100 * val_correct / val_total

        # Scheduler step
        scheduler.step(val_epoch_loss)

        # Check improvement
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"[{pool_mode}, {image_size}x{image_size}] Epoch {epoch+1}: Validation loss decreased. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"[{pool_mode}, {image_size}x{image_size}] Epoch {epoch+1}: No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= PATIENCE:
            print(f"[{pool_mode}, {image_size}x{image_size}] Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"\n[{pool_mode}, {image_size}x{image_size}] Training complete. Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}.")
    print(f"[{pool_mode}, {image_size}x{image_size}] Best model saved at {model_save_path}")

if __name__ == "__main__":
    full_df = load_dataset(CSV_PATH)
    print(f"Total samples after removing multi-labels: {len(full_df)}")

    train_df, val_df = train_test_split(
        full_df,
        test_size=VALIDATION_SPLIT,
        stratify=full_df['label'],
        random_state=RANDOM_SEED
    )
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    class_weights = compute_class_weights(train_df['label'].values, len(CLASSES))
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")

    # Define the exact sequence of trainings
    training_sequence = [
        ('global_average', 256)
        # ('max', 256),
        # ('second_order', 512),
        # ('max', 512),
        # ('second_order', 1024),
        # ('max', 1024)
    ]

    for pool_mode, img_size in training_sequence:
        print(f"\n\n=== Training model with {pool_mode} pooling at resolution {img_size}x{img_size} ===")

        if img_size != 256:
            BATCH_SIZE /= 2
            NUM_WORKERS /= 2

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = NIHChestXrayDataset(dataframe=train_df, image_dir=IMAGE_DIR, transform=train_transform)
        val_dataset = NIHChestXrayDataset(dataframe=val_df, image_dir=IMAGE_DIR, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        MODEL_SAVE_PATH = f"nih_xray_cnn_best_{pool_mode}_{img_size}.pth"
        train_model(train_loader, val_loader, class_weights, pool_mode, img_size, MODEL_SAVE_PATH)
