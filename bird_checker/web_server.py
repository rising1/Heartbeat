import io
import torchvision.transforms as transforms
from PIL import Image
from model import model_builder
import constants

model_builder.load_latest_saved_model(constants.BIRDIES_MODEL)
print("loading model .. " + constants.BIRDIES_MODEL )
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(96),
                                        transforms.CenterCrop(72),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)

    return model_builder.predict(tensor)


