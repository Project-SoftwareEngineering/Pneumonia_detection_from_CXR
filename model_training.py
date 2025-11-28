import torch

print(f"Is GPU available?  -> {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"How many GPUs?     -> {torch.cuda.device_count()}")
    print(f"Current GPU ID:    -> {torch.cuda.current_device()}")
    print(f"Current GPU name:  -> {torch.cuda.get_device_name(torch.cuda.current_device())}")
import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision.models as models
import torch.utils.data as data
from torch.utils.data import ConcatDataset, Subset
import torch.nn.functional as F

from torch.autograd import Variable
import sys
try:
    import clip
except ImportError:
    print("CLIP library not found. Please install with: pip install openai-clip")

# New imports for splitting and metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score
import seaborn as sns

from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter

# For plotting
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# ------------------ Original Augmentation Functions --------------
# -----------------------------------------------------------------

def add_g(image_array, mean=0.0, var=30):
    """Adds Gaussian noise to a NumPy array image."""
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    """Flips a NumPy array image horizontally."""
    return cv2.flip(image_array, 1)

# -----------------------------------------------------------------
# ------------------ Core Model & Loss (Unchanged) ----------------
# -----------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, drop_rate=0):
        super(Model, self).__init__()
        
        self.resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.fc_in_dim = self.resnet50.fc.in_features
        self.fc = nn.Linear(self.fc_in_dim, 512)
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x, text_features, clip_model, targets, phase='train'):
        with torch.no_grad():            
            image_features = clip_model.encode_image(x)
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        
        x_text = x  @ text_features.T
        x2 =  x * image_features  
        x_vision = self.fc_out(x2)

        a = 1
        b = 0.2
        out = a*x_vision + b*x_text

        if phase=='train':
            return out
        else:
            return out

# -----------------------------------------------------------------
# ------------------ Train/Validate/Test Functions ----------------
# -----------------------------------------------------------------

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def train(args, model, train_loader, optimizer, scheduler, device, text_features, clip_model):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    
    model.to(device)
    model.train()

    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

    for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(train_loader):
        imgs1 = imgs1.to(device)
        labels = labels.to(device)
    
        output = model(imgs1, text_features, clip_model, labels, phase='train')
        CE_loss = nn.CrossEntropyLoss()(output, labels)
        
        lsce_loss = lsce_criterion(output, labels)
        loss1 = 2 * lsce_loss + CE_loss
        loss = loss1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_cnt += 1
        _, predicts = torch.max(output, 1)
        correct_num = torch.eq(predicts, labels).sum()
        correct_sum += correct_num
        running_loss += loss

    scheduler.step()
    running_loss = running_loss / iter_cnt
    acc = correct_sum.float() / float(len(train_loader.dataset))
    return acc, running_loss

def validate(model, val_loader, device, text_features, clip_model):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0

        for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(val_loader):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)

            outputs = model(imgs1, text_features, clip_model, labels, phase='test')
            loss = nn.CrossEntropyLoss()(outputs, labels)

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            running_loss += loss
            data_num += outputs.size(0)

        running_loss = running_loss / iter_cnt
        val_acc = correct_sum.float() / float(data_num)
        
    return val_acc, running_loss

def test(model, test_loader, device, text_features, clip_model, class_names):
    # Load the best model saved during training
    model.load_state_dict(torch.load("./best_model/ours_best_Chest_2class_3_0.2text_1vision.pth")['model_state_dict'])
    
    with torch.no_grad():
        model.eval()
        
        all_labels = []
        all_preds = []

        for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(test_loader):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)
            outputs = model(imgs1, text_features, clip_model, labels, phase='test')
            
            _, predicts = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicts.cpu().numpy())

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    print("---" * 20)
    print(f"📊 FINAL TEST SET EVALUATION (NEW SPLIT) 📊")
    print("---" * 20)
    print(f"Total Test Images: {len(all_labels)}")
    print(f"Balanced Accuracy: {bal_acc * 100:.2f}%")
    print(f"Overall Accuracy:  {(np.diag(cm).sum() / cm.sum()) * 100:.2f}%")
    print(f"Weighted Precision:  {precision * 100:.2f}%")
    print(f"Weighted Recall:     {recall * 100:.2f}%")
    print(f"Weighted F1-Score:   {f1 * 100:.2f}%")
    
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (New Test Split)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# -----------------------------------------------------------------
# -------------------- Data Loader (with orig. augs) --------------
# -----------------------------------------------------------------

class ImageFolderDataset(data.Dataset):
    """
    Base class that uses a cv2_loader.
    We need this so the class below can inherit its cv2_loader.
    """
    def __init__(self, root, transform=None, phase='train', basic_aug=True):
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.image_folder = datasets.ImageFolder(root=root, loader=self.cv2_loader)

    def cv2_loader(self, path):
        """Loads an image with OpenCV and converts to RGB."""
        image = cv2.imread(path)
        return image[:, :, ::-1] # Convert BGR to RGB

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        image1 = flip_image(image)
        if self.transform is not None:
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = add_g(image)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx, image1

class ImageFolderDatasetFromSamples(ImageFolderDataset):
    """
    Creates a dataset from a list of (path, label) samples.
    Inherits the cv2_loader from ImageFolderDataset.
    """
    def __init__(self, samples, transform=None, phase='train', basic_aug=True):
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        self.samples = samples # List of (path, label)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.cv2_loader(path) # Use loader from parent class
        
        image1 = flip_image(image)
        if self.transform is not None:
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = add_g(image)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, idx, image1

# -----------------------------------------------------------------
# ------------------ Configuration & Setup ------------------------
# -----------------------------------------------------------------

# Set sharing strategy for num_workers > 0
torch.multiprocessing.set_sharing_strategy('file_system')

class Args:
    raf_path = 'chest_xray/' 
    
    resnet50_path = '../../resnet50_ft_weight.pkl'
    label_path = 'list_patition_label.txt'
    workers = 8      # Use more workers
    batch_size = 64  # Use larger batch size
    w = 7
    h = 7
    gpu = 0
    lam = 5.0
    epochs = 60
    # Splits are now hard-coded below: 70% train, 15% val, 15% test
    val_split_size = 0.15
    test_split_size = 0.15

args = Args()

# Set up GPU
torch.cuda.set_device(args.gpu)
device = torch.device(f'cuda:{args.gpu}')
print(f'Using device: {device}')

# Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Text features setup
label_to_emotion = {0: "NORMAL", 1: "PNEUMONIA"}
emotions = [f"a photo of a {label_to_emotion[i]}" for i in range(0, 2)]

with torch.no_grad():
    text_inputs = clip.tokenize(emotions).to(device)
    text_features = clip_model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.float().to(device)

# --- Transforms ---
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------------------------
# ------------------ NEW: Data Splitting (70/15/15) ---------------
# -----------------------------------------------------------------
print("Loading and splitting all data (train, val, test)...")
data_dir = args.raf_path

# 1. Load *all* data sources
train_data_source = datasets.ImageFolder(root=os.path.join(data_dir, 'train'))
val_data_source = datasets.ImageFolder(root=os.path.join(data_dir, 'val'))
test_data_source = datasets.ImageFolder(root=os.path.join(data_dir, 'test'))

# Get class names for metrics
class_names = train_data_source.classes
print(f"Class to index: {train_data_source.class_to_idx}")

# 2. Combine them into one giant pool
all_samples = train_data_source.samples + val_data_source.samples + test_data_source.samples
all_targets = train_data_source.targets + val_data_source.targets + test_data_source.targets
print(f"Total images found in all folders: {len(all_samples)}")

# 3. First split: 70% train, 30% temp (val+test)
train_samples, temp_samples, train_targets, temp_targets = train_test_split(
    all_samples,
    all_targets,
    test_size=(args.val_split_size + args.test_split_size), # 0.30
    stratify=all_targets,
    random_state=42
)

# 4. Second split: Split the 30% temp set 50/50
# (0.15 / 0.30 = 0.50)
relative_test_size = args.test_split_size / (args.val_split_size + args.test_split_size)
val_samples, test_samples = train_test_split(
    temp_samples,
    test_size=relative_test_size, # 0.50
    stratify=temp_targets,
    random_state=42
)

# 5. Create the final datasets with correct transforms
train_dataset = ImageFolderDatasetFromSamples(
    train_samples, transform=train_transforms, phase='train', basic_aug=True)

val_dataset = ImageFolderDatasetFromSamples(
    val_samples, transform=eval_transforms, phase='test', basic_aug=False)

test_dataset = ImageFolderDatasetFromSamples(
    test_samples, transform=eval_transforms, phase='test', basic_aug=False)

print(f"  -> New Train set size: {len(train_dataset)}")
print(f"  -> New Validation set size: {len(val_dataset)}")
print(f"  -> New Test set size: {len(test_dataset)}")


# --- Dataloaders ---
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.workers, pin_memory=True)
print("Dataloaders ready.")


# --- Model, Optimizer, Scheduler ---
model = Model(num_classes=2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

print("Setup complete. Ready to train.")

# -----------------------------------------------------------------
# ---------------------- Run Training -----------------------------
# -----------------------------------------------------------------

setup_seed(3407)
best_val_acc = 0

os.makedirs("./best_model/", exist_ok=True)
# Clear old results.txt file
if os.path.exists('results.txt'):
    os.remove('results.txt')

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [], 
    'val_acc': []
}

print("Starting training...")
for i in range(1, args.epochs + 1):
    train_acc, train_loss = train(args, model, train_loader, optimizer, scheduler, device, text_features, clip_model)
    val_acc, val_loss = validate(model, val_loader, device, text_features, clip_model)

    # Log metrics
    history['train_loss'].append(train_loss.item())
    history['train_acc'].append(train_acc.item())
    history['val_loss'].append(val_loss.item())
    history['val_acc'].append(val_acc.item())

    print(f"Epoch: {i:02}/{args.epochs} | "
          f"Train Loss: {train_loss.item():.4f} | Train Acc: {train_acc.item():.4f} | "
          f"Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"  -> New best model saved with val_acc: {best_val_acc:.4f}")
        torch.save({'model_state_dict': model.state_dict(),}, "./best_model/ours_best_Chest_2class_3_0.2text_1vision.pth")
    
    torch.save({'model_state_dict': model.state_dict(),}, "./best_model/ours_final_Chest_2class_0.2text_1vision.pth")
    
    with open('results.txt', 'a') as f:
        f.write(f"Epoch: {i}, Train_Acc: {train_acc.item():.4f}, Val_Acc: {val_acc.item():.4f}, Best_Val_Acc: {best_val_acc:.4f}\n")

print("Training finished.")
print(f"Best validation accuracy: {best_val_acc:.4f}")

# -----------------------------------------------------------------
# ---------------------- Plot Results -----------------------------
# -----------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot Accuracy
ax1.plot(history['train_acc'], label='Train Accuracy')
ax1.plot(history['val_acc'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Plot Loss
ax2.plot(history['train_loss'], label='Train Loss')
ax2.plot(history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------
# ---------------------- FINAL TEST -------------------------------
# -----------------------------------------------------------------

# Load the best model and run it on the new, fair test set
test(model, test_loader, device, text_features, clip_model, class_names)
