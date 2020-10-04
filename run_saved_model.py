import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import time
import os, csv
import glob


dataPathRoot = 'E:/'
loadfile = True


def load_latest_saved_model(chosen_model = None,is_eval = False):
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
        checkpoint = torch.load(comp_root +  selected_model,map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        #  model.train()
        if not is_eval:
            model_file_path = comp_root + selected_model
            interim_fig_prev_text = model_file_path[(model_file_path.rfind('_') + 1):(len(model_file_path) - 6)]
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
    first_learning_rate(optimizer,learning_rate)
    print("model loaded")
    return model


if __name__ == "__main__":

    # ------------------------------------------------------------------
    #  fixed prediction == labels.data,
    #-------------------------------------------------------------------
    # loaded_model = load_latest_saved_model()
    # loaded_model = load_latest_saved_model("new")
    # loaded_model = load_latest_saved_model("Birdies_model_110__best_38_loss_0.081436545.model")
    loaded_model = load_latest_saved_model("Birdies_model_0.model_best_acc_4.2667")
    # train(200)
    # View_Test.test(model,eval_loader, 'E:/bird_list.txt')