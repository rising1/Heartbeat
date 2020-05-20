import io, requests
import torchvision.transforms as transforms
from PIL import Image
import Hawknet_v2

Hawknet_v2.load_latest_saved_model(
    "Birdies_model_4__best_14_FDpsBSksFn_64_72_24_3_16.model")
print("loading model .. " + "Birdies_model_4__best_14_FDpsBSksFn_64_72_24_3_16.model" )
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
    # print("type=" + str(type(tensor)) + str(tensor))
    return Hawknet_v2.predict(tensor)


