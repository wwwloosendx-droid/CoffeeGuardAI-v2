import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
from datetime import datetime
import pickle
import json

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# DEVICE SETUP
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ============================================
# DATA AUGMENTATION
# ============================================
class CustomImageDataset(Dataset):
    """Custom Dataset for Coffee leaf images with augmentation"""
    
    def __init__(self, image_paths, labels, transform=None, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            return None
        
        # Resize
        img = cv2.resize(img, (224, 224))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        return img, label

# ============================================
# AUGMENTATION TRANSFORMS
# ============================================
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ============================================
# LOAD DATASET
# ============================================
def load_dataset(dataset_path):
    """Load images and labels from dataset directory"""
    logger.info(f"Loading dataset from {dataset_path}")
    
    image_paths = []
    labels = []
    label_map = {}
    label_id = 0
    
    for batch in sorted(os.listdir(dataset_path)):
        batch_path = os.path.join(dataset_path, batch)
        images_path = os.path.join(batch_path, "images")
        
        if not os.path.exists(images_path):
            logger.warning(f"Skipping {batch} (no images folder)")
            continue
        
        # Map batch name to label ID
        if batch not in label_map:
            label_map[batch] = label_id
            label_id += 1
        
        logger.info(f"Loading {batch}...")
        
        # Load images from this batch
        for img_name in os.listdir(images_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(images_path, img_name)
                image_paths.append(img_path)
                labels.append(label_map[batch])
    
    logger.info(f"Total samples loaded: {len(image_paths)}")
    logger.info(f"Class mapping: {label_map}")
    
    return image_paths, labels, label_map

# ============================================
# MODEL DEFINITION
# ============================================
class CoffeeLeafCNN(nn.Module):
    """Custom CNN for Coffee leaf disease classification"""
    
    def __init__(self, num_classes):
        super(CoffeeLeafCNN, self).__init__()
        
        # Use ResNet50 pretrained backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers for transfer learning
        for param in list(self.backbone.parameters())[:-50]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ============================================
# TRAINING FUNCTION
# ============================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# ============================================
# VALIDATION FUNCTION
# ============================================
def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_labels

# ============================================
# MAIN TRAINING LOOP
# ============================================
def main():
    logger.info("="*60)
    logger.info("STARTING CNN TRAINING FOR COFFEE LEAF CLASSIFICATION")
    logger.info("="*60)
    
    # Configuration
    dataset_path = r"C:\Users\hp\Desktop\Coffee and Cashew Nut Dataset\Coffee\Coffee"
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Load dataset
    image_paths, labels, label_map = load_dataset(dataset_path)
    
    if len(image_paths) == 0:
        logger.error("No images found. Check dataset structure!")
        return
    
    # Split dataset
    logger.info("Splitting dataset into train/val/test (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = CustomImageDataset(X_train, y_train, transform=train_transform, is_train=True)
    val_dataset = CustomImageDataset(X_val, y_val, transform=val_transform, is_train=False)
    test_dataset = CustomImageDataset(X_test, y_test, transform=val_transform, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"DataLoaders created with batch size: {batch_size}")
    
    # Model setup
    num_classes = len(label_map)
    model = CoffeeLeafCNN(num_classes).to(device)
    logger.info(f"Model created for {num_classes} classes")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_accuracy = 0
    patience = 10
    patience_counter = 0
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING STARTED")
    logger.info("="*60)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'coffee_cnn_best.pth')
            logger.info(f"  ✓ Best model saved! (Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience reached)")
            break
    
    # Load best model
    logger.info("\nLoading best model for testing...")
    model.load_state_dict(torch.load('coffee_cnn_best.pth'))
    
    # Test set evaluation
    logger.info("\n" + "="*60)
    logger.info("TEST SET EVALUATION")
    logger.info("="*60)
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Detailed metrics
    logger.info("\n--- Classification Report ---")
    class_names = list(label_map.keys())
    class_report = classification_report(test_labels, test_preds, target_names=class_names)
    logger.info(f"\n{class_report}")
    
    logger.info("\n--- Confusion Matrix ---")
    conf_matrix = confusion_matrix(test_labels, test_preds)
    logger.info(f"\n{conf_matrix}")
    
    # Save metadata
    metadata = {
        'num_classes': num_classes,
        'class_map': label_map,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'best_val_accuracy': best_val_accuracy,
        'model_architecture': 'ResNet50',
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('coffee_cnn_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info("\n" + "="*60)
    logger.info("✓ MODEL TRAINING COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    logger.info(f"Model saved as: coffee_cnn_best.pth")
    logger.info(f"Metadata saved as: coffee_cnn_metadata.json")
    logger.info(f"Logs saved as: training.log")

if __name__ == "__main__":
    main()
