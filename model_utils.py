import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import clip
import os

class IntegratedModel(nn.Module):
    def __init__(self, num_classes=2):
        super(IntegratedModel, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.fc_in_dim = self.resnet50.fc.in_features
        self.fc = nn.Linear(self.fc_in_dim, 512)
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x, text_features, clip_model):
        with torch.no_grad():
            image_features = clip_model.encode_image(x)
        
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.fc(x)
        
        x_text = x @ text_features.T
        x2 = x * image_features
        x_vision = self.fc_out(x2)
        
        a, b = 1.0, 0.2
        out = a*x_vision + b*x_text
        return out

class SystemBackend:
    def __init__(self, model_path='models/ours_best_Chest_2class_3_0.2text_1vision.pth'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.load_error = None
        print(f"Loading System on: {self.device}...")
        
        try:
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            
            labels = ["NORMAL", "PNEUMONIA"]
            emotions = [f"a photo of a {l}" for l in labels]
            text_inputs = clip.tokenize(emotions).to(self.device)
            with torch.no_grad():
                self.text_features = self.clip_model.encode_text(text_inputs)
                self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
                self.text_features = self.text_features.float()

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            self.model = IntegratedModel(num_classes=2)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            print("✅ Trained Model Loaded Successfully.")

        except Exception as e:
            self.model_loaded = False
            self.load_error = str(e)
            print(f"❌ CRITICAL ERROR: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def check_blur(self, image_np, threshold=100):
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            return variance, variance < threshold
        except:
            return 0.0, False

    def predict(self, image_pil):
        if not self.model_loaded:
            raise RuntimeError(f"AI Model not loaded: {self.load_error}")

        try:
            img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor, self.text_features, self.clip_model)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probs, 1)
                
            label = "PNEUMONIA" if predicted_class.item() == 1 else "NORMAL"
            return label, confidence.item() * 100
            
        except Exception as e:
            raise RuntimeError(f"Inference Failed: {str(e)}")