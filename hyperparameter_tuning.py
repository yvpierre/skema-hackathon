"""
Grid search for optimal hyperparameters
"""
import itertools
from fine_tune_models import ImprovedCNN, fine_tune_with_frozen_layers
from improved_training import train_improved_model, get_improved_hyperparameters

def hyperparameter_search():
    """Try different hyperparameter combinations"""
    
    # Define search space
    learning_rates = [0.0001, 0.00005, 0.001]
    batch_sizes = [8, 16, 32]
    model_names = ['resnet50', 'efficientnet_b0', 'densenet121']
    
    results = []
    
    for lr, bs, model_name in itertools.product(learning_rates, batch_sizes, model_names):
        print(f"\n{'='*60}")
        print(f"Testing: LR={lr}, BS={bs}, Model={model_name}")
        print(f"{'='*60}")
        
        # Create model
        model = ImprovedCNN(num_classes=2, model_name=model_name, pretrained=True)
        model = fine_tune_with_frozen_layers(model, freeze_until_layer=-20)
        
        # Modify hyperparameters
        import improved_training
        params = get_improved_hyperparameters()
        params['lr'] = lr
        params['batch_size'] = bs
        
        # Train (you'll need to modify train_improved_model to accept params)
        try:
            trained_model = train_improved_model(model)
            # Evaluate and store results
            # results.append({'lr': lr, 'bs': bs, 'model': model_name, 'acc': val_acc})
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Print best combination
    # best = max(results, key=lambda x: x['acc'])
    # print(f"\n🏆 Best combination: {best}")

if __name__ == '__main__':
    hyperparameter_search()
