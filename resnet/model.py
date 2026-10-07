'''
3.1. model.py
    - Contains the ResNetClassifier class
    - Loads a ResNet-18 backbone and replaces the final fully-connected layer to match the desired number of classes.

'''
import torch

class ResNetClassifier(torch.nn.Module):
    """
    A ResNet-18 based classifier with a customizable number of output classes.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
