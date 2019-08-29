import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
import ConvNet
import HawkDataLoader
from torch.optim import Adam
import os, csv
import glob
import time

global faff, snapshot_points, batch_sizes, \
    loadfile, dataPathRoot, model, \
    optimizer, loss_fn, pic_size, epoch, \
    loss, device, train_loader_class, \
    learning_rate, test_loader, \
    single_loader_class, num_epochs

def build_model():
    global dataPathRoot, faff, snapshot_points, \
            batch_sizes, loadfile, model, optimizer, \
            loss_fn
    # Hyperparameters
    colour_channels = 3  # used in SimpleNet
    no_feature_detectors = 12  # used in ??????
    kernel_sizes = 3  # used in Unit
    stride_pixels = 1  # used in Unit
    padding_pixels = 1  # used in Unit
    pooling_factor = 2  # used in SimpleNet
    pic_size = 72  # used in SimpleNet
    output_classes = 220  # used in SimpleNet
    learning_rate = 0.001  # used in HeartbeatClean
    weight_decay = 0.0001  # used in HeartbeatClean
    dropout_factor = 0.1  # used in Unit
    faff = 'false'
    num_epochs = 300  # used in HeartbeatClean
    snapshot_points = num_epochs / 1
    batch_sizes = 256  # used in HeartbeatClean
    #  batch_sizes = 6 # used in HeartbeatClean
    loadfile = True

    #  dataPathRoot = 'F:/BirdiesData/' # used in DataLoaderHeartbeat
    #  dataPathRoot = 'C:/Users/phfro/Documents/python/data/BirdiesData/' # used in DataLoaderHeartbeat
    #  if not (os.path.exists(dataPathRoot)):
    #  dataPathRoot = 'C:/Users/peter.frost/Documents/python/data/BirdiesData/'  # used in DataLoaderHeartbeat
    #  dataPathRoot = '/content/drive/My Drive/Colab Notebooks/BirdiesData'
    dataPathRoot = '/content/drive/My Drive/Colab Notebooks'
    print("parameters loaded and data root path set")

    SimpleNetArgs = [kernel_sizes, stride_pixels, padding_pixels, dropout_factor,
                 output_classes, colour_channels, pic_size, pooling_factor]
    model = ConvNet.SimpleNet(SimpleNetArgs)
    optimizer = Adam(model.parameters(), lr=learning_rate,
                 weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    print("model now built")


def get_latest_file(path, *paths):
    """Returns the name of the latest (most recent) file
    of the joined path(s)"""
    fullpath = os.path.join(path, *paths)
    list_of_files = glob.glob(fullpath)  # You may use iglob in Python3
    if not list_of_files:                # I prefer using the negation
        return None                      # because it behaves like a shortcut
    latest_file = max(list_of_files, key=os.path.getctime)
    _, filename = os.path.split(latest_file)
    return filename

def transfer_to_gpu():
    global device
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    print("device=", device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.to(device)
        print("model transferred to GPU")

def load_latest_saved_model(chosen_model = None):
    global dataPathRoot, loadfile, model, optimizer, \
            epoch, loss, device
    # load a saved model if one exists
    comp_root = dataPathRoot + "/saved_models/"


    if chosen_model is not None:
        selected_model = chosen_model
        print("looking for ",comp_root + selected_model)
        print("exists = ",os.path.isfile(comp_root + selected_model))
    else:
        stub_name = "Birdies_model_*"
        selected_model = get_latest_file(comp_root, stub_name)
        print("latest filename=", selected_model)

    if os.path.isfile(comp_root + selected_model) and loadfile == True:
        checkpoint = torch.load(comp_root +  selected_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.train()
        model_file_path = comp_root + get_latest_file(comp_root, stub_name)
        interim_fig_prev_text = model_file_path[(model_file_path.rfind('_') + 1):(len(model_file_path) - 1)]
        interim_fig_prev = float(interim_fig_prev_text)
        print("using saved model ", model_file_path, " Loss: {:.4f}".format(interim_fig_prev))
    else:
        print("using new model")
    #  finished deciding where the model comes from

    #  For the given model

    #  Print model's state_dict
    #  print("Model's state_dict:")
    #  for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        if var_name == "param_groups":
            print(var_name, "\t", optimizer.state_dict()[var_name])
    print("model loaded")

def set_up_training(is_training):

    global model, train_loader_class, val_loader_class, \
            test_loader_class, single_loader_class, train_loader, \
            val_loader, test_loader

    if(is_training):
        model.train()
        train_loader_class = \
            HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
        val_loader_class = \
            HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
        test_loader_class = \
            HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
        single_loader_class = \
            HawkDataLoader.HawkLoader(dataPathRoot, batch_sizes, pic_size)
        train_loader = train_loader_class.dataloaders["train"]
        val_loader = val_loader_class.dataloaders["val"]
        test_loader = test_loader_class.dataloaders["test"]
        # Get a batch of training data
        inputs, classes = next(iter(train_loader))
        print('len inputs=', len(inputs))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        print("model in training mode")
    else:
        model.eval()
        print("model in evaluation mode")

def save_models(epoch, loss, save_point):
    save_PATH = dataPathRoot + "/saved_models/" + "Birdies_model_{}.model".format(epoch) + save_point
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }
    torch.save(checkpoint, save_PATH)
    print("Checkpoint saved")
    if (os.path.exists(save_PATH)):
        print("verified save ", save_PATH)


def train(num_epochs):
    #  global variables used in this function
    global interim_fig_prev

    print("In train")

    #  set some measurement variables
    best_acc = 0.0
    since = time.time()
    train_history = []
    loopcount = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
                print('model set to train mode')
            else:
                model.eval()   # Set model to evaluate mode
                print('model set to eval mode')

            running_loss = 0.0
            running_corrects = 0
            batch_counter = 0

            # Iterate over data.
            for inputs, labels in train_loader_class.dataloaders[phase]:

                #  set up reporting variables
                no_of_batches = int(train_loader_class.dataset_sizes[phase] / batch_sizes) + 1
                batch_counter = batch_counter + 1

                if batch_counter == 1:
                    print("inputs size=",inputs.shape)
                    print("labels size=",labels.shape)
                print('Epoch=',epoch,' batch=',batch_counter," of ",no_of_batches)

                #  push the data to the GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                print(' prev loss.item()=', loss.item(), ' x inputs.size(0)=', inputs.size(0))
                running_corrects += torch.sum(preds == labels.data)
                time_elapsed = time.time() - since
                interim_fig = running_loss / ((epoch + 1) * batch_counter)
                interim_corrects = running_corrects / ((epoch + 1) * batch_counter)
                #  if batch_counter == 1:
                #    interim_fig_prev = interim_fig
                #    save_models(epoch, loss, "_loss_{:.4f} ".format(interim_fig))
                #    print("first save ", "_loss_{:.4f} ".format(interim_fig))
                print( phase, " Running_loss: {:.4f}, Average_loss: {:.4f}, Running_corrects: {:.4f},"
                      .format(running_loss, interim_fig,
                              interim_corrects), 'time {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                print("Average_loss: {:.4f},Prev_average_loss: {:.4f}, Learning_rate: {:.7f}".format(
                        interim_fig, interim_fig_prev, learning_rate))
                if interim_fig < interim_fig_prev:
                    interim_fig_prev = interim_fig
                    interim = "_loss_{:.4f} ".format(running_loss / ((epoch + 1) * batch_counter))
                    print("saving at ",interim)
                    save_models(epoch, loss, interim)

            train_loss = running_loss / train_loader_class.dataset_sizes[phase]
            train_acc = running_corrects.double() / \
                        train_loader_class.dataset_sizes[phase]


            # Evaluate on the test set
            test_acc = test_train()

            # Save the model if the test acc is greater than our current best
            if test_acc > best_acc:
                main_acc = "_best_acc_{:.4f} ".format(test_acc)
                save_models(epoch,loss,main_acc)
                best_acc = test_acc
                print("best accuracy= ", best_acc)

        if ((epoch) % (num_epochs / snapshot_points) == 0) or (epoch == num_epochs):
            loopcount = loopcount + 1
            time_elapsed = time.time() - since
            print("Epoch {:4}, ".format(epoch),
                 phase, " Accuracy: {:.4f},TrainLoss: {:.4f},"
                  .format( train_acc,
                          train_loss), 'time {:.0f}m {:.0f}s'.format(
                          time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))
            # Accuracy Curves
            train_history.append(train_acc)


def test_train():
    model.eval()
    test_acct = 0.0
    test_history = []
    images, labels = next(iter(test_loader))

    if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

    #  Predict classes using images from the test set
    outputs = model(images)
    _, prediction = torch.max(outputs.data, 1)

    test_acct += torch.sum(prediction == labels.data)
    #  print("test_acct= ",  (test_acct).cpu().numpy())
    #  Compute the average acc and loss over all 10000 test images
    test_acct = test_acct.cpu().numpy() / 30
    test_history.append(test_acct)
    #  print("in test")
    return test_acct

def test():
    model.eval()
    test_acct = 0.0
    test_history = []
    images, labels = next(iter(test_loader))

    if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

    #  Predict classes using images from the test set
    outputs = model(images)
    _, prediction = torch.max(outputs.data, 1)
    print("prediction=",single_loader_class.classes[int(prediction.cpu().numpy())])

def test_single(images):
    image_list = []
    guess_list = []
    for image in images:
        #  Predict classes using images from the test set
        image = image[0].to(device)
        img = image.cpu()
        inp = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        image_list.append(inp)
        outputs = model(image.unsqueeze(0))
        _, prediction = torch.max(outputs.data, 1)
        guess = birds_listing()[int(prediction.cpu().numpy())]
        guess_list.append(guess)
        #  imshow(img, guess)
        #  print("prediction=",guess)
    show_images(image_list,2,guess_list)


def birds_listing():
    with open('/content/drive/My Drive/Colab Notebooks/Class_validate.txt', 'r') as f:
       reader = csv.reader(f)
       classes = list(reader)[0]
       classes.sort()
       #  self.classes = open('/content/drive/My Drive/Colab Notebooks/Class_validate.txt').read()
       #  print("self.classes=",classes)
       #  print("len self.classes=",len(classes))
    return classes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated


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
        a.set_title(title)
    #  fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig = plt.figure(figsize=(6, 3))
    plt.show()

# train(num_epochs)
if __name__ == "__main__":
      build_model()
      transfer_to_gpu()
      load_latest_saved_model()
      set_up_training()
      train(num_epochs)
    #test()
