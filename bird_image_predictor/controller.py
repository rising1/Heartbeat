from app import app
from flask import render_template, request
from werkzeug.utils import secure_filename
import os

from bird_image_predictor import image_handler, view_test

rootdir = 'f:/'
validate_path = 'f:/bird_list.txt'

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
    if request.method == 'POST':
        fileob = request.files['file2upload']
        filename = secure_filename(fileob.filename)
        save_pathname = os.path.join(
            app.config['UPLOAD_FOLDER'],filename)
        fileob.save(save_pathname)

        identified = image_handler.handle()
    return identified

app.config["UPLOAD_FOLDER"] =  "f:/uploads"

@app.route('/test')
def test():
    return render_template('test.html')