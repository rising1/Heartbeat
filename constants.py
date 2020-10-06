import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RESOURCES_ROOT = os.path.join(PROJECT_ROOT, 'resources')
SAVED_MODELS = os.path.join(PROJECT_ROOT, 'saved_models')
BIRD_LIST = os.path.join(RESOURCES_ROOT, 'bird_list.txt')
BIRDIES_MODEL = os.path.join(SAVED_MODELS, "Birdies_model_4__best_14_FDpsBSksFn_64_72_24_3_16.model")
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.0001