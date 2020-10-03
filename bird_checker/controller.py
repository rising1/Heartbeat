from app import app
from flask import render_template, request
from werkzeug.utils import secure_filename
import os

from bird_checker import web_server, view_test


validate_path = './Class_validate.txt'

@app.route('/')

@app.route('/index')
def index():
    user = {'username': 'Pete'}
    return render_template('index.html',title='Home',  user=user)

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET','POST'])
def uploader_file():
    print("in uploader")
    if request.method == 'POST':
        print("in post in uploader")
        fileob = request.files['file2upload']
        print(fileob.filename)
        filename = secure_filename(fileob.filename)
        save_pathname = os.path.join(
            app.config['UPLOAD_FOLDER'],filename)
        fileob.save(save_pathname)
        with open(save_pathname, 'rb') as f_bytes:
            print("save pathname=" + save_pathname)
            image_bytes = f_bytes.read()
            prediction_number = int(web_server.get_prediction(
                image_bytes).cpu().numpy())
            # print("prediction number=" + str(prediction_number))
            identified = view_test.birds_listing(
                validate_path)[prediction_number]
            print(identified)

    return identified

app.config["UPLOAD_FOLDER"] =  "./temp"

@app.route('/test')
def test():
    return render_template('test.html')