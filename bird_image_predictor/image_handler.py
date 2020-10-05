import io
import torchvision.transforms as transforms
from PIL import Image
from model import model_builder
import constants
import numpy as np

from bird_image_predictor import view_test

#TODO:// move to do this on start of app
model_builder.load_and_populate_model(constants.BIRDIES_MODEL)
# print("loading model .. " + constants.BIRDIES_MODEL )



def handle(filepath):
    choiceslist = []
    with open(filepath, 'rb') as f_bytes:
        image_bytes = f_bytes.read()
        scores, predictedplaces = _get_prediction(
            image_bytes)
        # print("prediction number=" + str(prediction_number))
        for i in predictedplaces:
            # print(i)
            choiceslist.append(view_test.birds_listing(
                constants.BIRD_LIST)[i])
        for j in scores:
            choiceslist.append(" (score " + str(np.round(j, 2)) + ")")
    return choiceslist

def _transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(96),
                                        transforms.CenterCrop(72),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def _get_prediction(image_bytes):
    tensor = _transform_image(image_bytes=image_bytes)
    # print("type=" + str(type(tensor)) + str(tensor))
    return model_builder.predict(tensor)


