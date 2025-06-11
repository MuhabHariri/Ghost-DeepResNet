import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
from src.config import *
from src.data_pipeline import build_dataset
from src.model_blocks import Ghost_DeepResNet_Model
import time
from tqdm import tqdm

class SaveWeightsPerEpoch:
    def __init__(self, output_dir="Model_weights", model_name="model_weights"):
        self.output_dir = output_dir
        self.model_name = model_name
        os.makedirs(self.output_dir, exist_ok=True)

    def save_weights(self, model, epoch):
        file_path = os.path.join(self.output_dir, f"{self.model_name}_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), file_path)
        print(f"✅ Saved weights at: {file_path}")

class EarlyStopping:
    def __init__(self, patience=15, monitor='val_accuracy', restore_best_weights=True, verbose=1):
        self.patience = patience
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

    def restore_best_weights_to_model(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels_idx = torch.max(labels.data, 1)  
        total += labels.size(0)
        correct += (predicted == labels_idx).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels_idx = torch.max(labels.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels_idx).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def run_training():
    print("Checking GPU availability...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"PyTorch Version: {torch.__version__}")

    #load File Paths & Build Datasets
    print("Preparing datasets...")
    train_file_paths = glob.glob(os.path.join(TRAIN_DIR, "*", "*"))
    val_file_paths = glob.glob(os.path.join(VAL_DIR, "*", "*"))
    test_file_paths = glob.glob(os.path.join(Test_DIR, "*", "*"))

    train_dataloader = build_dataset(train_file_paths)
    val_dataloader = build_dataset(val_file_paths)
    test_dataloader = build_dataset(test_file_paths)

    train_samples = len(train_file_paths)
    val_samples = len(val_file_paths)
    test_samples = len(test_file_paths)
    train_steps_per_epoch = len(train_dataloader)
    val_steps_per_epoch = len(val_dataloader)

    print(f"Train samples: {train_samples} | Val samples: {val_samples} | "
          f"Train steps/epoch: {train_steps_per_epoch} | Val steps/epoch: {val_steps_per_epoch}")

    #model Definition
    model = Ghost_DeepResNet_Model(
        num_classes=NUM_CLASSES,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        stages_config=STAGES
    )
    
    #multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)

    #Print model summary
    print("\nModel Architecture:")
    print(model)
    
    #count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    #Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()  

    #Callbacks
    early_stopping = EarlyStopping(
        patience=PATIENCE,
        monitor='val_accuracy',
        restore_best_weights=True,
        verbose=1
    )
    
    save_weights_callback = SaveWeightsPerEpoch(
        output_dir="Model_weights", 
        model_name="final_model"
    )

    # Training Loop
    print("\nStarting training...")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 50)
        
        #training phase
        train_loss, train_acc = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        
        #validation phase
        val_loss, val_acc = validate_one_epoch(model, val_dataloader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        #save weights every epoch
        save_weights_callback.save_weights(model, epoch)
        
        #early stopping check
        early_stopping(val_acc, model)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")
        
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    #restore best weights
    if early_stopping.restore_best_weights:
        early_stopping.restore_best_weights_to_model(model)
        print("Restored best weights")

    #Save final model
    os.makedirs("models", exist_ok=True)
    
    #Save the entire model
    torch.save(model, "models/final_classifier_model.pth")
    
    #Save just the state dict 
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), "models/final_classifier_model_state_dict.pth")
    else:
        torch.save(model.state_dict(), "models/final_classifier_model_state_dict.pth")
    
    print("Model training complete and saved to models/")
    
    #Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate_one_epoch(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

