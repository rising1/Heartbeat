#import Categorize
#import Load_pix_and_clean
#import CleanDirectory
#import Hawknet_Depld
#src_data = '/content/drive/My Drive/Colab Notebooks/train'
#categorize = Categorize.Categorize('/content/drive/My Drive/Colab Notebooks/train',
#               '/content/drive/My Drive/Colab Notebooks/train')
#categorize = Categorize.Categorize('/content/drive/My Drive/Colab Notebooks/',
#               '/content/drive/My Drive/Colab Notebooks/')
#clean_directory = CleanDirectory.CleanDirectories(
#                  '/content/drive/My Drive/Colab Notebooks/train')
#categorize.copy_top_up_images()
#  categorize.summarise()
#file_list = categorize.get_more_images()
#loc_data = '/content/drive/My Drive/Colab Notebooks/trial'
#print(file_list[0])
#load_pix = Load_pix_and_clean.Load_pix(file_list,loc_data,src_data)
# !python "Categorize.py"

import HawkNet
import Hawknet_Depld
validate_path = 'C:/Users/phfro/Documents/python/data/Class_validate.txt'
dataPathRoot = 'C:/Users/phfro/Documents/python/data'

from flask import Flask
app = Flask(__name__)
@app.route("/")
def hello():
        return "Hello World!"

HawkNet.build_model(dataPathRoot)
HawkNet.transfer_to_gpu()
is_eval = True
HawkNet.load_latest_saved_model('Birdies_model_0.model_best_acc_4.2667_',is_eval)
#  HawkNet.load_latest_saved_model()
#  HawkNet.load_latest_saved_model("New")

HawkNet.set_up_training(False)
#HawkNet.train(50)
deploy_test = Hawknet_Depld.test_images(
        'C:/Users/phfro/Documents/python/data/eval/' )
#  HawkNet.show_images(deploy_test.test_image)
#  HawkNet.imshow(deploy_test.test_images) --> 28.08.19
HawkNet.test_single(deploy_test.test_images, validate_path)
#  /content/drive/My Drive/Colab Notebooks/saved_models/Birdies_model_0.model_best_acc_4.2667

if __name__ == '__main__':
        app.run(debug=True)