import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Disease labels common in chest X-ray datasets
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
    "Mass", "Nodule", "Pneumonia", "Pneumothorax"
]

class GradCAMDenseNet:
    def __init__(self):
        # 1. Load pretrained DenseNet121 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # Modify classifier head for target pathologies
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, len(DISEASE_LABELS))
        
        self.model.to(self.device)
        self.model.eval()

        # Placeholders for gradients and activations
        self.gradients = None
        self.activations = None

        # Hook the final convolutional layer of DenseNet121
        target_layer = self.model.features.denseblock4
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def predict_and_explain(self, image_path, output_gradcam_path):
        orig_image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(orig_image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_()

        # Forward pass
        output = self.model(input_tensor)
        probabilities = torch.sigmoid(output).squeeze(0)

        # Get top predicted disease
        pred_idx = torch.argmax(probabilities).item()
        predicted_disease = DISEASE_LABELS[pred_idx]
        confidence = probabilities[pred_idx].item()

        # Backward pass for Grad-CAM
        self.model.zero_grad()
        target_score = output[0, pred_idx]
        target_score.backward()

        # Calculate Grad-CAM weights and map
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        # Overlay heatmap on original image
        orig_cv = cv2.imread(image_path)
        h, w, _ = orig_cv.shape
        cam_resized = cv2.resize(cam, (w, h))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        cam_overlay = cv2.addWeighted(orig_cv, 0.6, heatmap, 0.4, 0)

        cv2.imwrite(output_gradcam_path, cam_overlay)

        return {
            "predicted_disease": predicted_disease,
            "confidence": round(confidence * 100, 2),
            "gradcam_path": output_gradcam_path
        }

if __name__ == "__main__":
    detector = GradCAMDenseNet()
    print("DenseNet121 + Grad-CAM pipeline initialized successfully!")