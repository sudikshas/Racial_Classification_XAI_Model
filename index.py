from flask import Flask, render_template, request, redirect, flash, url_for
import main
import urllib.request
from app import app
from werkzeug.utils import secure_filename
from main import getPrediction
import os


@app.route('/')
def index():
    print("inside index")
    return render_template('index.html')


@app.route('/', methods=['POST'])
def submit_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
        #getPrediction(filename)
        label, acc = getPrediction(filename)
        # render_template('index.html', filename=filename)
        flash(label)
        flash(acc)
        flash(filename)
        flash('Image successfully uploaded and displayed')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

    # if request.method == 'POST':
    #     if 'file' not in request.files:
    #         flash('No file part')
    #         return redirect(request.url)
    #     file = request.files['file']
    #     if file.filename == '':
    #         flash('No file selected for uploading')
    #         return redirect(request.url)
    #     if file:
    #         filename = secure_filename(file.filename)
    #         file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    #         getPrediction(filename)
    #         label, acc = getPrediction(filename)
    #         # render_template('index.html', filename=filename)
    #         flash(label)
    #         flash(acc)
    #         flash(filename)
    #         flash('Image successfully uploaded and displayed')
    #         #return render_template('index.html', filename=filename)

    #         return redirect('/')

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
    print("inside display_image")
    #return redirect(url_for("static", filename=filename), code=301)


if __name__ == "__main__":
    app.run()