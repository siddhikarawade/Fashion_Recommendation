from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from preprocess import batch_preprocess
from recommendation_engine import recommend_from_image

UPLOAD_FOLDER = 'static/uploads'
PREPROCESS_INPUT = 'preprocess_input'
PREPROCESS_OUTPUT = 'static/preprocessed'

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin', methods=['GET'])
def admin():
    return render_template('admin.html')

@app.route('/admin/preprocess', methods=['POST'])
def start_preprocessing():
    msg = batch_preprocess(PREPROCESS_INPUT, PREPROCESS_OUTPUT)
    flash(msg)
    return redirect(url_for('admin'))

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("No file selected")
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        recommendations = recommend_from_image(filepath)
        return render_template('results.html', input_img=filename, recommendations=recommendations)

@app.route('/camera', methods=['GET'])
def camera():
    # Displays camera capture interface (see camera.html)
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(debug=True)

