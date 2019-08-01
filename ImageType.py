from torchvision import models
import torch

print(dir(models))
resnet = models.resnet101(pretrained=True)
print(resnet)