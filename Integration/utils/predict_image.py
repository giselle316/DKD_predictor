import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 多层FC的Cre模型
class CrescentModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CrescentModel, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# 单层FC的Fib模型
class FiberosisModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FiberosisModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    def forward(self, x):
        return self.model(x)

def load_model(path, model_type='crescent'):
    if model_type == 'crescent':
        model = CrescentModel()
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    elif model_type == 'fiberosis':
        model = FiberosisModel()
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise ValueError("model_type 必须是 'crescent' 或 'fiberosis'")
    model.eval()
    return model

# 预处理函数
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_crescent(image_path):
    model_path = 'models/Crescent.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 未找到")
    model = load_model(model_path, model_type='crescent')
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return int(predicted.item()), float(confidence.item())

def predict_fibrosis(image_path):
    model_path = 'models/Fiberosis.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 未找到")
    model = load_model(model_path, model_type='fiberosis')
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return int(predicted.item()), float(confidence.item())
