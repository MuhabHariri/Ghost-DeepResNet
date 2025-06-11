import os
import glob
import random
import pathlib
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from src.config import *
from src.config import CLASS_NAMES

import warnings
warnings.filterwarnings('ignore')

# CLASS_NAMES = np.array([
#     item.name for item in pathlib.Path(TRAIN_DIR).glob('*') if item.name != "LICENSE.txt"
# ])

def get_label(file_path):
    """Extract label from file path"""
    parts = file_path.split(os.path.sep)
    folder_name = parts[-2]
    # Convert to one-hot encoding like TensorFlow version
    return (CLASS_NAMES == folder_name).astype(np.float32)

def load_and_preprocess_image(path):
    """Load and preprocess image"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SHAPE),  # Assuming IMAGE_SHAPE is (height, width)
        transforms.ToTensor(),  # This automatically converts to [0,1] range and changes to CHW format
    ])
    
    # Load image
    image = Image.open(path).convert('RGB')  # Ensure 3 channels
    image = transform(image)
    return image

def load_and_preprocess_data(path):
    """Load and preprocess both image and label"""
    image = load_and_preprocess_image(path)
    label = get_label(path)
    return image, label

class CustomDataset(Dataset):
    """Custom PyTorch Dataset class"""
    def __init__(self, file_paths):
        self.file_paths = file_paths
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image, label = load_and_preprocess_data(path)
        return image, torch.tensor(label, dtype=torch.float32)

def build_dataset(file_paths, shuffle_buffer=5000):
    """Build PyTorch DataLoader equivalent to TensorFlow dataset"""
    # Shuffle file paths
    random.shuffle(file_paths)
    dataset = CustomDataset(file_paths)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  
        num_workers=4,  
        pin_memory=True,  
        drop_last=False
    )
    
    return dataloader

def build_dataset_with_buffer_shuffle(file_paths, shuffle_buffer=5000):
    from torch.utils.data.sampler import SubsetRandomSampler
    
    # Shuffle file paths
    random.shuffle(file_paths)
    
    # Create dataset
    dataset = CustomDataset(file_paths)
    
    # Create indices for shuffling within buffer
    indices = list(range(len(dataset)))
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(indices),  # Random sampling
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader

