import io
import torchvision.transforms as transforms
from PIL import Image
from model import model_builder
import constants

from bird_image_predictor import view_test

#TODO:// move to do this on start of app
model_builder.load_and_populate_model(constants.BIRDIES_MODEL)
# print("loading model .. " + constants.BIRDIES_MODEL )
def handle(filepath):
    with open(filepath, 'rb') as f_bytes:
        image_bytes = f_bytes.read()
        prediction_number = int(_get_prediction(
            image_bytes).cpu().numpy())
        # print("prediction number=" + str(prediction_number))
        identified = view_test.birds_listing(
            constants.BIRD_LIST)[prediction_number]
        return identified

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


