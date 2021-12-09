from flask import Blueprint, render_template, request
from werkzeug.utils import secure_filename
from static.torch_model.u2net import U2NETP
import random
import os
import numpy
from static.torch_model.toolof_ocrdetect import *
"""
切换CPU与GPU 需注释
"""
# 注册蓝图
ocr_vn = Blueprint("ocr_vn", __name__)
# 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']
# 保存文件地址
UPLOAD_FOLDER_OCR_VN = './static/images/ocr_detect'
# 模型参数路径
pth_data_1 = './static/torch_model/ocr_detect_model.pth'
pth_data_2 = './static/torch_model/ocr_read_model.pth'
# 模型实例化
model_1 = U2NETP().cpu()
model_2 = U2NETP().cpu()
# 模型初始化
model_1.eval()
model_2.eval()
# model_1.cuda()  # 切换GPU1
model_1.load_state_dict(torch.load(pth_data_1, map_location=torch.device('cpu')))  # 将参数加载到模型里(CPU)
model_2.load_state_dict(torch.load(pth_data_2, map_location=torch.device('cpu')))  # 将参数加载到模型里(CPU)
# model_1.load_state_dict(torch.load(pth_data_1))  # 切换GPU2
# model_2.load_state_dict(torch.load(pth_data_2))  # 切换GPU3


def after_request(resp):
    """
    跨域支持
    :param resp:响应
    :return:
    """
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


ocr_vn.after_request(after_request)


def allowed_file(filename):
    """
    # 判断文件后缀是否在列表中
    :param filename:
    :return:
    """
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS


def contours_get(img0):
    """
    图片后处理
    :param img0:
    :return:
    """
    img0 = numpy.array(img0 * 255, numpy.uint8)
    # 转换为灰度图
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # 将图片二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # 在二值图上寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img0, contours


# 以下是接口设置
# OCR检测(POST请求)
@ocr_vn.route("/ocr_detect", methods=['POST', 'GET'])
def detect():
    """
    模型检测接口，正常图片识别返回状态码200，并返回识别结果
    传入图片不符合要求或者其他错误返回状态码500
    :return: json响应
    """
    if request.method == 'POST':
        # 获取post过来的文件名称，从name=file参数中获取
        file = request.files['file']
        # 检测文件格式
        if file and allowed_file(file.filename):
            # secure_filename方法会去掉文件名中的中文
            try:
                file_name = secure_filename(file.filename)
                # 保存图片
                file.save(os.path.join(UPLOAD_FOLDER_OCR_VN, file_name))
                img_orgin_cv, pil_img = behind_img(os.path.join(UPLOAD_FOLDER_OCR_VN, file_name))
                # img_orgin_pil = Image.open(os.path.join(UPLOAD_FOLDER, file_name))
                # img_orgin_cv = cv2.imread(os.path.join(UPLOAD_FOLDER, file_name))
                inputs_test = tf(pil_img, 416, 416)
                # if torch.cuda.is_available():
                #     inputs_test = inputs_test.cuda()
                # 将数据传如模型 得到7个tensor张量（1，1，416，416）
                pred0, pred1, pred2, pred3, pred4, pred5, pred6 = model_1(inputs_test)
                # 将张量反算成（3，416，416）
                pred_mask = torch.cat((pred0, pred1, pred2), 1)[0]
                pred_mask = pred_mask
                # 将张量转化为numpy格式
                pred_mask = pred_mask.cpu().data.numpy()
                # 将c，h，w转换成h，w，c
                pred_mask = np.transpose(pred_mask, (1, 2, 0))
                # cv2.imshow('pred',pred_mask)
                # cv2.waitKey(0)
                boxs = mask2value(pred_mask)  # 得到外接矩形boxs

                img_0, img_2, img_7, img_x = get_cut_img1(img_orgin_cv, boxs)  # 得到想得到的数字切片
                img_0 = img_trans(img_0, (120, 120))
                pred_idnumber = model_2(img_0)
                img_2 = img_trans(img_2, (120, 120))
                pred_birth = model_2(img_2)
                img_7 = img_trans(img_7, (120, 120))
                pred_expire = model_2(img_7)
                # 身份证号码
                pred_idnumber = mask_read2value(ten2cv(pred_idnumber))
                # print(len(str(pred_idnumber).strip()))
                print(pred_idnumber)
                # 生日日期
                pred_birth = mask_read2value(ten2cv(pred_birth))
                print(pred_birth)
                # 到期时间
                pred_expire = mask_read2value(ten2cv(pred_expire))
                print(pred_expire)
                # 三个值检测不到视为识别失败
                if pred_idnumber.strip() == '' and pred_expire.strip() == '' and pred_birth.strip() == '':
                    return {"code": '500',
                            'message': '识别失败',
                            'result': {}}
                # 保存检测图片
                pred_mask1, contours = contours_get(pred_mask)
                pred_mask = pred_mask1.copy()
                for cont in contours:
                    # 取轮廓长度的1%为epsilon
                    epsilon = 0.005 * cv2.arcLength(cont, True)
                    # 预测多边形
                    box = cv2.approxPolyDP(cont, epsilon, True)
                    img = cv2.polylines(pred_mask, [box], True, (0, 0, 255), 2)
                cv2.drawContours(pred_mask, contours, -1, (100, 100, 100))
                # 检测后的图片保存（看上线需求）
                # imgpath = 'static/images/ocr_detect/{}{}'.format(random.randint(1000, 9999), file_name)
                # cv2.imwrite(imgpath, pred_mask1)
                return {"code": '200',
                        # "previewpath": 'static/images/ocr_detect/' + file_name,
                        # "resultpath": imgpath,
                        "message": "上传成功",
                        'result': {
                            'idnumber': pred_idnumber.strip(),
                            'birth': pred_birth.strip(),
                            'expire': pred_expire.strip(),
                            'name': "",
                            'place': '',
                            'address': "",
                            'sex': "",
                            'nation': ""},
                        }
            except(Exception):
                return {"code": '500',
                        'message': '识别失败',
                        'result': {}}
        else:
            return {"code": 404,
                    "message": "格式错误，仅支持jpg、png、jpeg格式文件",
                    "result": {}
                    }
    # 上线时注释
    # else:
    #     return {"code": '503',
    #             "message": "仅支持post方法", "result": {}}
    return render_template('ocr_vn.html')
