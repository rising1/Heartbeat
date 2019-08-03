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




class ImageType:

    model = squeezenet1_1(pretrained=True)
    model.eval()
    x = []

    def __init__(self,image_path):
        self.image_path = image_path
        index_file = "class_map.txt"



        indexpath = os.path.join(os.getcwd(), index_file)
        # Donwload class index if it doesn't exist
        if not os.path.exists(indexpath):
            print("cant find image indeces")
        with open(indexpath, "r") as f:
            self.class_map = f.readlines()
        print("type of self.class_map", type(self.class_map))

        # run prediction function annd obtain predicted class index
        index = self.predict_image()
        prediction = self.class_map[index]
        print("Predicted Class ", prediction)

    def predict_image(self):

        print("Prediction in progress")
        image = Image.open(self.image_path)

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
        output = self.model(input)

        index = output.data.numpy().argmax()

        return index


if __name__ == "__main__":
    ImageType("H:/birdiesdata2/Accentor/Alpine_1. 550px-prunella_collaris_collaris%2c_picos_de_europa.jpg")

