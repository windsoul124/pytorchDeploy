from flask import Blueprint, render_template, request
from werkzeug.utils import secure_filename

# 新建蓝图
face_recognition_blueprint = Blueprint("face_recognition", __name__)


@face_recognition_blueprint.route('/face_recognition')
def index():
    return render_template('face_recognition.html')



