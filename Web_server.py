from flask import Flask
app = Flask(__name__)

''' to run flask:
go to conda prompt
Set FLASK_ENV=development
Set FLASK_APP=Web_server.py
flask run
'''
import io
import torchvision.transforms as transforms
from PIL import Image
import Hawknet_v2


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

with open("F:/exp/woodpigeon/6ea941e5d1.jpg", 'rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    #print(tensor)

def get_prediction(image_bytes):
    Hawknet_v2.predict(image_bytes)

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return str(type(tensor))

