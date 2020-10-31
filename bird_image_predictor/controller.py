from app import app
from flask import render_template, request
from werkzeug.utils import secure_filename
import os


from bird_image_predictor import image_handler, setup

logging = app.logger

@app.route('/')

@app.route('/index')
def index():
    logging.info("/index endpoint hit")
    user = {'username': 'Pete'}
    return render_template('answer.html',title='Home',  user=user)

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods = ['GET','POST'])
def uploader_file():
    if request.method == 'POST':
        logging.info("Received bird picture")
        fileob = request.files['file2upload']
        filename = secure_filename(fileob.filename)
        save_pathname = os.path.join(
            app.config['UPLOAD_FOLDER'],filename)
        fileob.save(save_pathname)

        choiceslist = image_handler.handle(save_pathname, logging)
    logging.info("Bird predictions are: " + choiceslist[0])
    return choiceslist[0] + "," + choiceslist[3] + "," + \
           choiceslist[1] + "," + choiceslist[4] + "," + choiceslist[2] + "," + choiceslist[5]

app.config["UPLOAD_FOLDER"] =  "./temp"

@app.route('/test')
def test():
    return render_template('test.html')