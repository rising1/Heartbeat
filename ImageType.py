# Import needed packages
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import requests
import shutil
from io import open
import os
from PIL import Image
import json


model = squeezenet1_1(pretrained=True)
model.eval()


def predict_image(image_path):
    print("Prediction in progress")
    image = Image.open(image_path)

    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)

    index = output.data.numpy().argmax()

    return index


if __name__ == "__main__":

    index_file = "class_map.txt"

    indexpath = os.path.join(os.getcwd(), index_file)
    # Donwload class index if it doesn't exist
    if not os.path.exists(indexpath):
        print("cant find image indeces")

    class_map = json.load(open(indexpath))

    # run prediction function annd obtain prediccted class index
    index = predict_image(imagepath)

    prediction = class_map[str(index)][1]

    print("Predicted Class ", prediction)
