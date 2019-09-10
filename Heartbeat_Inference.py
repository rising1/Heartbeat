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
from flask import Flask, request, render_template

host = 'home'

if host == 'home':
    validate_path = 'C:/Users/phfro/Documents/python/data/Class_validate.txt'
    dataPathRoot = 'C:/Users/phfro/Documents/python/data'
    validate_path = 'C:/Users/phfro/Documents/python/data/Class_validate.txt'

if host ==  'work':
    validate_path = 'C:/Users/peter.frost/Downloads/Class_validate.txt'
    dataPathRoot = 'C:/Users/peter.frost/Documents/python/data/birdiesdata'
    test_image = 'C:/Users/peter.frost/Documents/python/data/birdiesdata/eval/'

app_route = '/'

HawkNet.build_model(dataPathRoot)
HawkNet.transfer_to_gpu(False)
is_eval = True
HawkNet.load_latest_saved_model('Birdies_model_0.model_best_acc_4.2667_',is_eval)
#  HawkNet.load_latest_saved_model()
#  HawkNet.load_latest_saved_model("New")

HawkNet.set_up_training(False)

app = Flask(__name__)
print("app.root_path=",app.root_path)
print("app.instance_path=",app.instance_path)
#  @app.route('/',methods=['GET','POST'])
@app.route(app_route,methods=['GET','POST'])
def hello():
        if request.method == 'GET':
                return render_template('index.html', value='hello')
        if request.method == 'POST':
            print(request.files)
            if 'file' not in request.files:
                print("file not uploaded")
                return
            file = request.files['file']
            image = file.read()
            deploy_test = Hawknet_Depld.test_images(image)
            predicted_bird = HawkNet.test_single(deploy_test.test_images, validate_path)
            return render_template('result.html', bird=predicted_bird)



#HawkNet.train(50)
# deploy_test = Hawknet_Depld.test_images(test_image )
# HawkNet.test_single(deploy_test.test_images, validate_path)


if __name__ == '__main__':
        app.run(debug=True)