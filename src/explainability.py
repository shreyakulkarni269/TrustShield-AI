"""
Explainability Module:
Implements Grad-CAM using PyTorch native forward/backward hooks to visualize
where EfficientNet-B0 looks when detecting manipulation artifacts.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import load_trained_model

class NativeGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Target the single binary logit
        output.backward(gradient=torch.ones_like(output))

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # ReLU on CAM
        cam = F.relu(cam)
        # Normalize between 0 and 1
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input resolution (224x224)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam.squeeze().detach().cpu().numpy()

def explain_image(image_path, model_path="./models/best_model.pth", output_path="./results/gradcam_sample.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_trained_model(model_path, device=device)

    # EfficientNet-B0 final feature extraction layer
    target_layer = model.features[-1]
    grad_cam = NativeGradCAM(model, target_layer)

    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logit = model(input_tensor)
        prob = torch.sigmoid(logit).item()

    # Heatmap
    heatmap = grad_cam.generate_heatmap(input_tensor)

    # Plot
    img_resized = img.resize((224, 224))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_resized)
    axes[0].set_title(f"Input Face (Score: {prob*100:.1f}%)")
    axes[0].axis('off')

    axes[1].imshow(img_resized)
    axes[1].imshow(heatmap, cmap='jet', alpha=0.55)
    axes[1].set_title("Grad-CAM Saliency Overlay")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved Grad-CAM visualization to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        explain_image(sys.argv[1])
    else:
        print("Usage: python src/explainability.py <path_to_image>")
