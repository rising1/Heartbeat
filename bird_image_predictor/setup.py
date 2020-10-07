import torch
import os
import glob
from torch.optim import Adam

import constants
from model import model_builder

loadfile = True
optimizer = Adam(model_builder.model.parameters(), lr=constants.LEARNING_RATE, weight_decay=constants.WEIGHT_DECAY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_populate_model(chosen_model = None, is_eval = False):
    #TODO:// Error handling to check model path exists
    global dataPathRoot, loadfile, model, optimizer, \
            epoch, loss, device
    # load a saved model if one exists
    comp_root = constants.BIRDIES_MODEL
    print("comp_root=" + comp_root)

    if chosen_model is not None:
        selected_model = chosen_model
        print("looking for ", constants.BIRDIES_MODEL)
        print("exists = ",constants.BIRDIES_MODEL)
    else:
        stub_name = "Birdies_model_*"
        selected_model = get_latest_file(constants.BIRDIES_MODEL)
        print("latest filename=", selected_model)

    if os.path.isfile(constants.BIRDIES_MODEL) and loadfile == True:
        checkpoint = torch.load(constants.BIRDIES_MODEL,map_location='cpu')
        model_builder.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        #  model.train()
        if not is_eval:
            model_file_path = constants.BIRDIES_MODEL
            interim_fig_prev_text = model_file_path[(model_file_path.rfind('_') + 1):(len(model_file_path) - 6)]
            interim_fig_prev = float(interim_fig_prev_text)
            print("using saved model ", model_file_path, " Loss: {:.4f}".format(interim_fig_prev))
    else:
        print("using new model")

    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        if var_name == "param_groups":
            print(var_name, "\t", optimizer.state_dict()[var_name])
    first_learning_rate(optimizer,constants.LEARNING_RATE)
    print("model loaded")
    return model_builder.model

def first_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("learning rate adjusted to ", lr)

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

load_and_populate_model(constants.BIRDIES_MODEL)

# TODO:// are these method important? and is it appropriate to be in this file? or back in model_builder

# def adjust_learning_rate(epoch,lr):
#
#     global num_epochs
#     if epoch == 8 * num_epochs / 10:
#         lr = lr / 2
#     elif epoch == 6 * num_epochs / 10:
#         lr = lr / 2
#     elif epoch == 4 * num_epochs / 10:
#         lr = lr / 2
#     elif epoch == 2 * num_epochs / 10:
#         lr = lr / 2
#
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr

# def save_models(epoch, loss, save_point):
#     print("save path types = ",str(type(dataPathRoot))+"\t",str(type(epoch))+"\t",str(type(save_point)))
#     save_PATH = dataPathRoot + "/saved_models/" + "Birdies_model_{}_".format(epoch) + "_best_" \
#                                 + str(save_point) + "_FDpsBSksFn_" + str(no_feature_detectors) + "_" +\
#                 str(pic_size) + "_" + str(batch_size) + "_" + str(kernel_sizes) + "_" + str(flattener) +".model"
#     checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             }
#     torch.save(checkpoint, save_PATH)
#     print("Checkpoint saved")
#     if (os.path.exists(save_PATH)):
#         print("verified save ", save_PATH)

