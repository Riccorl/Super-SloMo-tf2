import os

from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename

import config
import inference

UPLOAD_FOLDER = '/content/Super-SloMo-tf2/super-slomo/uploads'
PREDICT_FOLDER = '/content/Super-SloMo-tf2/super-slomo/predicted'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTED_FOLDER'] = PREDICT_FOLDER

run_with_ngrok(app)  # Start ngrok when app is run

model_path = "/content/drive/My Drive/Magistrale/Computer Vision/models/run11/chckpnt/ckpt-35"


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['PREDICTED_FOLDER'], filename)

            inference.predict_from_web(video_path, output_path, model_path)

            return redirect(url_for('predicted_file', filename=filename))
    else:
        return render_template("home.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/predicted/<filename>')
def predicted_file(filename):
    return send_from_directory(app.config['PREDICTED_FOLDER'],
                               filename)


def check():
    upload_path = os.path.join(config.CODE_DIR, "uploads")
    if not os.path.exists(upload_path):
        os.mkdir(upload_path)
    predicted_path = os.path.join(config.CODE_DIR, "predicted")
    if not os.path.exists(predicted_path):
        os.mkdir(predicted_path)


if __name__ == "__main__":
    check()
    app.run()
