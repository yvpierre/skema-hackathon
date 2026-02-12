"""
Enhanced data augmentation for better accuracy
"""
from torchvision import transforms
from torchvision.transforms import v2

def get_strong_augmentation():
    """Strong augmentation pipeline for industrial defect detection"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Added
        transforms.RandomRotation(degrees=30),  # Increased from 15
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Random shifts
            scale=(0.8, 1.2),      # Random zoom
            shear=10               # Random shear
        ),
        transforms.ColorJitter(
            brightness=0.3,  # Increased from 0.2
            contrast=0.3,
            saturation=0.2,  # Added
            hue=0.1          # Added
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Added
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),   # Added
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def get_test_transform():
    """Simple transform for validation/test"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

# For even stronger augmentation (if accuracy is very low)
def get_heavy_augmentation():
    """Heavy augmentation with RandomErasing and more"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomRotation(degrees=45),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.7, 1.3),
            shear=15
        ),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # Simulate occlusions
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
