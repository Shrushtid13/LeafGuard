import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

def continue_training(data_dir, model_path, epochs=1, batch_size=32):
    print("--- Step 3b: Resuming Training for Final Accuracy Boost ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    
    # Setup Model
    model = models.mobilenet_v3_large()
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['val'])

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer (Lower LR for refinement)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    criterion = nn.CrossEntropyLoss()

    # Training Loop (1 Epoch)
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in tqdm(dataloader, desc="Refining"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # Eval
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Final Eval"):
            inputs, labels = inputs.to(device), labels.to(device)
            # TTA-lite
            out = (model(inputs) + model(inputs.flip(-1))) / 2.0
            probs = torch.softmax(out, dim=1)
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    y_test_bin = label_binarize(all_labels, classes=range(len(class_names)))
    auc_score = roc_auc_score(y_test_bin, all_probs, multi_class='ovr')
    print(f"Refined ROC-AUC: {auc_score:.4f}")

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'report': report,
        'auc': auc_score
    }, model_path)
    print(f"Refined Model saved to {model_path}")

if __name__ == "__main__":
    continue_training(r"d:\Kaggle\cassava_leaf_disease_classification\processed", "main/cassava_model.pth")
