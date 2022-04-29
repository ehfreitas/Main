from flask import Flask, flash, redirect, url_for, render_template, request
import urllib.request
from werkzeug.utils import secure_filename
import joblib
import os
import cnn_module
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'static/uploads'

app = Flask(__name__)
app.secret_key = 'secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def upload_form():
   return render_template('dogbreed.html')
	
@app.route('/', methods = ['POST'])
def upload_image():
    file = request.files['file']
    filename = secure_filename(file.filename)    
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    inceptionv3_model = load_model('../models/inceptionv3_model.h5')
    prediction = cnn_module.detect_human_dog(os.path.join(app.config['UPLOAD_FOLDER'], filename), inceptionv3_model)
    print(prediction)
    return render_template('dogbreed.html', filename=filename, prediction=prediction)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
   app.run(debug = True)