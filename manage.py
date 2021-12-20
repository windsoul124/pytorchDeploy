from flask import Flask, render_template
from service import ocr_vn_id, face_comparison, face_recognition


app = Flask(__name__)
# 注册OCR检测
app.register_blueprint(ocr_vn_id.ocr_vn)
# 人脸比对
app.register_blueprint(face_comparison.face_comparison_blueprint)
# 人脸识别
# app.register_blueprint(face_recognition.face_recognition_blueprint)


# 首页
# @app.route('/')
# def index():
#     return render_template('index.html')


# 处理中文编码
app.config['JSON_AS_ASCII'] = False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
