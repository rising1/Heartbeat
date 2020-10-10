from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch
import numpy as np
import csv
#import hawknet_depld
import constants
# dataPathRoot = 'C:/Users/phfro/PycharmProjects/Heartbeat'
# validate_path = 'C:/Users/phfro/PycharmProjects/Heartbeat/bird_list.txt'
dataPathRoot = 'f:/'
validate_path = constants.BIRD_LIST

def test(model,my_test_loader,validate_path):
    bird_list = [ 'Bittern', 'Blackbird',  'Chicken', 'Dove','Sparrowhawk', 'Owl','Parakeet', 'Peregrine', 'Plover',
                 'Puffin', 'Robin', 'Tit']
    my_test_loader_eval = my_test_loader
    model.eval()
    test_acct = 0.0
    test_history = []
    image_list = []
    label_list = []
    predictions_list = []
    images, labels = next(iter(my_test_loader_eval))
    if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
    #  Predict classes using images from the test set
    outputs = model(images)
    _, prediction = torch.max(outputs.data, 1)
    # probs_val = torch.max(outputs.data).item()

    for image in images:
        image = image.cpu()
        image_list.append(imshow(image))
    for i in range(len(prediction)):
        if (birds_listing(validate_path)[int(prediction[i].cpu().numpy())]) == (
                bird_list[labels.data[i].cpu().numpy()]):
                tick = "Y " + str("{:.1f}".format(_[i].cpu().numpy()))    # str(u'\2714'.encode('utf-8')) # approval tick mark
        else:
                tick = "No" + str("{:.1f}".format(_[i].cpu().numpy()))    # str(u'\2717'.encode('utf-8')) # cross mark
        predictions_list.append(birds_listing(validate_path)[int(prediction[i].cpu().numpy())]  +
                                # "\n" + "\n" + "       " + tick)
                                 "\n" + bird_list[labels.data[i].cpu().numpy()] + " " + tick)
    show_images(image_list,2,predictions_list)

def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title,fontsize=8)
    #  fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    #  fig = plt.figure(figsize=(6, 3))
    plt.show(block=False)
    plt.pause(120)
    plt.close()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    image_list = []
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def birds_listing(validate_path):
    with open(validate_path,'r') as f:
       #  with open('C:/Users/phfro/Documents/python/data/bird_list.txt', 'r') as f:
       #  with open('/content/drive/My Drive/Colab Notebooks/bird_list.txt', 'r') as f:
       reader = csv.reader(f)
       classes = list(reader)[0]
       classes.sort()
       #  self.classes = open('/content/drive/My Drive/Colab Notebooks/bird_list.txt').read()
       #  print("self.classes=",classes)
       #  print("len self.classes=",len(classes))
    return classes


