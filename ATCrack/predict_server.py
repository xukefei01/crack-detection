import json
from flask import Flask,jsonify, request,redirect,url_for,render_template,flash
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import time
from predict import *

from datetime import timedelta

app = Flask(__name__)

UPLOAD_FOLDER = 'upload/images'  # 上传的文件保存目录

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}  # 允许的上传文件类型

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=5)
print(app.config['SEND_FILE_MAX_AGE_DEFAULT'])

model = predict_init()


def allow_filename(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
    # 文件名必须有效


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


@app.route('/upload/images', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('not file part!')
            return redirect(request.url)

        f = request.files['file']
        if f.filename == '':
            flash('not file upload')
            return redirect(request.url)

        if f and allow_filename(f.filename):
            filename = secure_filename(f.filename)
            # secure_filename 不支持中文文件名称的获取…

            filepath = './' + app.config['UPLOAD_FOLDER'] + '/' + f.filename
            print(filepath)
            f.save(filepath)
            img = predict_segment(model, filepath)
            return render_template('upload_ok.html')
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2222, debug=True)