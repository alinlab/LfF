import torch.nn as nn
from module.resnet import resnet20
from module.mlp import MLP
from torchvision.models import resnet18, resnet50

def get_model(model_tag, num_classes):
    if model_tag == "ResNet20":
        return resnet20(num_classes)
    elif model_tag == "ResNet18":
        model = resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, num_classes)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        return model
    elif model_tag == "ResNet50":
        model = resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, num_classes)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        return model
    elif model_tag == "MLP":
        return MLP(num_classes=num_classes)
    else:
        raise NotImplementedError
