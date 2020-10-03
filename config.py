import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'no-way-jose'
    UPLOAD_FOLDER = '.'
