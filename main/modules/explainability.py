import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def save_activations(module, input, output):
            self.activations = output
            
        target_layer.register_forward_hook(save_activations)
        target_layer.register_full_backward_hook(save_gradients)

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        output[0, class_idx].backward()
        
        gradients = self.gradients
        activations = self.activations
        
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= np.max(heatmap)
        return heatmap

def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    return np.uint8(255 * superimposed_img / np.max(superimposed_img))

def run_explainability(model_path, test_data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    
    model = models.mobilenet_v3_large()
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Generating Explainability (Grad-CAM) Visualizations...")
    for cls in class_names:
        cls_path = os.path.join(test_data_dir, cls)
        if not os.path.exists(cls_path): continue
        
        img_name = os.listdir(cls_path)[0]
        img_path = os.path.join(cls_path, img_name)
        img = Image.open(img_path).convert('RGB')
        
        input_tensor = transform(img).unsqueeze(0)
        heatmap = grad_cam.generate_heatmap(input_tensor, None)
        
        # Overlay
        img_np = np.array(img.resize((224, 224)))
        overlay = overlay_heatmap(img_np, heatmap)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title(f"Original: {cls}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM Attention")
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, f"explain_{cls}.png"))
        plt.close()
    
    print(f"✅ Explainability reports saved to: {save_dir}")

if __name__ == "__main__":
    # This will be run AFTER train.py finishes
    model_path = "main/cassava_model.pth"
    test_dir = r"d:\Kaggle\cassava_leaf_disease_classification\processed\test"
    run_explainability(model_path, test_dir, "main/explain_plots")
