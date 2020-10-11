import io
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import numpy as np

from model.model_builder import model, cuda_avail
import constants
from bird_image_predictor import view_test

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
            choiceslist.append( str(np.round(j, 2)) )
        # print("choiceslist=" + str(choiceslist))
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
    model.eval()
    if cuda_avail:
        tensor = Variable(tensor.cuda())
    outputs = model(tensor)
    birdrank = (outputs.data).cpu().numpy()
    birdrank.flatten
    birdvalrank = np.flip(np.sort(birdrank), 1)
    firstchoice = np.where(birdrank == birdvalrank[0][0])
    secondchoice = np.where(birdrank == birdvalrank[0][1])
    thirdchoice = np.where(birdrank == birdvalrank[0][2])
    descendingchoices = []
    descendingscores = []
    # print("birdrank length=" + str(len(birdrank[0])))
    # print("birdrank shape=" + str(birdrank[0]))
    # print("birdvalrank length=" + str(len(birdvalrank[0])))
    for i in range(len(birdrank[0])):
        descendingchoices.append(np.where(birdrank == birdvalrank[0][i]))
    print("descendingchoices=" + str(descendingchoices))
        # descendingscores.append(float(birdvalrank[0][i]) + 100)

    scores = [float(birdvalrank[0][0]) + 100, float(birdvalrank[0][1]) + 100, float(birdvalrank[0][2]) + 100]
    print(str(scores))
    rankings = [int(firstchoice[1]), int(secondchoice[1]), int(thirdchoice[1])]
    print(str(rankings))
    return scores, rankings


