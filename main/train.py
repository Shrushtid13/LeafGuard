import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

def train_model(data_dir, model_save_path="cassava_model.pth", epochs=1, batch_size=32):
    print("--- Step 3: Final Performance Optimization (88%+ Strategy) ---")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'tta': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 100% Training Data
    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['val'])

    dataloaders = {
        'train': DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    }
    
    dataset_sizes = {'train': len(full_train_dataset), 'val': len(val_dataset)}
    class_names = full_train_dataset.classes
    
    # Class Weights for final push
    class_counts = [0] * len(class_names)
    for _, label in full_train_dataset.samples:
        class_counts[label] += 1
    total_train = sum(class_counts)
    class_weights = torch.tensor([total_train / c for c in class_counts], dtype=torch.float).to(device)
    class_weights = class_weights / class_weights.min()

    # Upgraded Model: MobileNetV3_Large (Powerful yet fast on CPU)
    model = models.mobilenet_v3_large(weights='DEFAULT')
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n--- Final Project Training Loop ---")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in tqdm(dataloaders[phase], desc=phase, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train': scheduler.step()
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            history[f'{phase}_loss'].append(epoch_loss); history[f'{phase}_acc'].append(epoch_acc.item())
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    
    # --- Step 4: Final Evaluation with TTA (Test Time Augmentation) ---
    print("\n--- Final Evaluation on Test Set (with TTA) ---")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc="Eval with TTA"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Simple TTA: Average original and one augmented pass
            out1 = model(inputs)
            # Re-apply transform for TTA (simulated)
            out2 = model(inputs.flip(-1)) # Horizontal flip TTA
            
            outputs = (out1 + out2) / 2.0
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute Metrics
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nFinal Classification Report:")
    print(report)
    
    # ROC-AUC Calculation
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(all_labels, classes=range(len(class_names)))
    auc_score = roc_auc_score(y_test_bin, all_probs, multi_class='ovr')
    print(f"\nFinal ROC-AUC Score: {auc_score:.4f}")
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1); plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val'); plt.title('Accuracy History'); plt.legend()
    plt.subplot(1, 2, 2); sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues'); plt.title('Final Confusion Matrix')
    plt.savefig('main/training_results.png')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'history': history,
        'report': report
    }, "main/" + model_save_path)
    print(f"Final Model saved to main/{model_save_path}")

if __name__ == "__main__":
    data_path = r"d:\Kaggle\cassava_leaf_disease_classification\processed"
    if os.path.exists(data_path):
        train_model(data_dir=data_path, epochs=1) 
    else:
        print("Processed data not found.")
