# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from chatterbot.input import InputAdapter
from chatterbot.conversation import Statement
import os
import logging
import flask
import time
import datetime
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import webbrowser
import psutil
import subprocess
import exifutil
import psycopg2
from config import Config
import cStringIO as StringIO
from flask import json
from PIL import Image
import base64
from chatterbot import ChatBot
UPLOAD_FOLDER = '/home/reena-mary/Pictures/caffe-demos-uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])
userhome = os.path.expanduser('~')

chatresponse=""
app = flask.Flask(__name__)

bot = ChatBot("Reena",
    storage_adapter="chatterbot.storage.JsonFileStorageAdapter",
    logic_adapters=[
        "chatterbot.logic.MathematicalEvaluation",
        "chatterbot.logic.BestMatch"
    ],
    output_adapter="chatterbot.output.TerminalAdapter",
    database="database.db"
    )

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

@app.route('/alina_chat', methods=['GET'])
def alina_chat():
    chatrequest_o = flask.request.args.get('chatrequest', '')
    # Create a new chat bot named Charlie
    # Get a response to the input text 'How are you?'
    
    i = "'{0}'"
    chatrequest = i.format(chatrequest_o)
    
    try:
        print(chatrequest)
        result = bot.get_response(chatrequest)
    except:
        return flask.render_template(
            'index.html', has_result=False,
            result='Exception in alina_chat.')
    
        
    
    return flask.render_template(
         'index.html', has_result=True, result=result, chatrequest_o=chatrequest_o)

@app.route('/find_name', methods=['GET'])
def find_name():
    # Create a new chat bot named Charlie
    # Get a response to the input text 'How are you?'
    
    filename = "/home/reena-mary/caffe-face-caffe-face/python/output.txt"
    with open(filename, 'r+') as f:
        nameresult = f.read()
        f.seek(0)
        f.truncate()
    
    return flask.render_template(
         'index.html', has_result_name=True, nameresult=nameresult, chatrequest_o=chatrequest_o)

@app.route('/face_identify', methods=['POST'])
def face_identify():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        imagepath = os.path.join(UPLOAD_FOLDER, filename_)
        #print("-------------------------------------------------------")
        #print(filename)
        imagefile.save(imagepath)
        logging.info('Saving to %s.', imagepath)
        image = exifutil.open_oriented_im(imagepath)
        


    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=False,
            result=(False, 'Cannot open uploaded image.')
        )
    
    start_caffe(imagepath)
    
    #return flask.render_template('index.html', has_result=True, result=result, imagesrc=embed_image_html(image), predsrc=yolopred)
    return flask.render_template(
        'index.html', has_result_face=True, imgsrc=embed_image_pred(image))



def embed_image_pred(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil2 = Image.fromarray((255 * image).astype('uint8'))
    #image_pil2 = image_pil.resize((256, 256))
    string_buf2 = StringIO.StringIO()
    image_pil2.save(string_buf2, format='png')
    data = string_buf2.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def start_caffe(imagepath):
    req_path = userhome + "/caffe-face-caffe-face/"
    os.chdir(req_path)
    cmd = "python python/classify.py --print_results %s foo" % imagepath
    # Added by Reena Mary
    s = subprocess.Popen([cmd],shell=True)
    p = psutil.Process(s.pid)
    try:
        p.wait(timeout=10)
    except psutil.TimeoutExpired:
        p.kill()
        return 0
    return 0   


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    #open_url()
    webbrowser.get('google-chrome').open_new_tab("0.0.0.0:"+str(port))
    #start_yolo()
    tornado.ioloop.IOLoop.instance().start()
    
    
	
def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)
   
   

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
