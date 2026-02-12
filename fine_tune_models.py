"""
Fine-tune pre-trained models for better accuracy
This approach typically gives 10-30% higher accuracy than training from scratch
"""
import torch
import torch.nn as nn
from torchvision import models
from improved_training import train_improved_model

class ImprovedCNN(nn.Module):
    """Fine-tuned pre-trained model - much better than BaselineCNN"""
    
    def __init__(self, num_classes=2, model_name='resnet50', pretrained=True):
        super(ImprovedCNN, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            # Replace final layer
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)

def fine_tune_with_frozen_layers(model, freeze_until_layer=-20):
    """
    Freeze early layers, only train last layers
    This prevents overfitting on small datasets
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last layers
    layers = list(model.modules())
    for layer in layers[freeze_until_layer:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    return model

if __name__ == '__main__':
    # Method 1: Fine-tune ResNet50 (recommended)
    print("🚀 Training with ResNet50 backbone (fine-tuning)...")
    model = ImprovedCNN(num_classes=2, model_name='resnet50', pretrained=True)
    
    # Freeze early layers (only train last 20 layers)
    model = fine_tune_with_frozen_layers(model, freeze_until_layer=-20)
    
    # Train
    trained_model = train_improved_model(model)
    
    # Save
    torch.save(trained_model.state_dict(), './models/improved_cnn_resnet50.pth')
    print("✅ Model saved to ./models/improved_cnn_resnet50.pth")
    
    # Method 2: Try EfficientNet (best for small datasets)
    # Uncomment to try:
    # print("\n🚀 Training with EfficientNet-B0 backbone...")
    # model = ImprovedCNN(num_classes=2, model_name='efficientnet_b0', pretrained=True)
    # model = fine_tune_with_frozen_layers(model, freeze_until_layer=-15)
    # trained_model = train_improved_model(model)
    # torch.save(trained_model.state_dict(), './models/improved_cnn_efficientnet.pth')
