"""
Improved training script with better hyperparameters and techniques
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from improved_augmentation import get_strong_augmentation, get_test_transform

def get_class_weights(dataset):
    """Calculate class weights for imbalanced datasets"""
    targets = [label for _, label in dataset.samples]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(targets),
        y=targets
    )
    return torch.FloatTensor(class_weights)

def create_weighted_sampler(dataset):
    """Create weighted sampler for balanced batches"""
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def get_improved_hyperparameters():
    """Return improved hyperparameters"""
    return {
        'batch_size': 16,  # Smaller for better gradient estimates
        'epochs': 100,     # More epochs with early stopping
        'lr': 0.0001,      # Lower learning rate
        'weight_decay': 1e-4,  # L2 regularization
        'patience': 15,    # Early stopping patience
        'lr_patience': 5,  # ReduceLROnPlateau patience
        'lr_factor': 0.5,  # LR reduction factor
    }

def train_improved_model(model, train_dir='./data/train', val_split=0.2):
    """
    Train model with improved settings
    
    Args:
        model: PyTorch model to train
        train_dir: Path to training data
        val_split: Validation split ratio
    """
    # Hyperparameters
    params = get_improved_hyperparameters()
    
    # Load datasets with strong augmentation
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=get_strong_augmentation()
    )
    
    # Split train/val
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply test transform to validation
    val_subset.dataset = datasets.ImageFolder(
        train_dir,
        transform=get_test_transform()
    )
    
    # Create weighted sampler for balanced training on the subset
    targets = [train_dataset.samples[idx][1] for idx in train_subset.indices]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=params['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss with class weights
    class_weights = get_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize validation accuracy
        factor=params['lr_factor'],
        patience=params['lr_patience'],
        verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(params['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{params["epochs"]}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), './models/baseline_cnn_best.pth')
            print(f'✅ New best model saved! Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
            print(f'⚠️  No improvement for {patience_counter} epochs')
            
            if patience_counter >= params['patience']:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    print(f'\n🎉 Training complete! Best Val Acc: {best_val_acc:.2f}%')
    return model

if __name__ == '__main__':
    # Example usage
    from streamlit_app import BaselineCNN  # Import your model
    
    model = BaselineCNN(num_classes=2)
    trained_model = train_improved_model(model)
