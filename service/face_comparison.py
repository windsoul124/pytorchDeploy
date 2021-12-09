from flask import Blueprint, render_template, request
from werkzeug.utils import secure_filename
import os
from skimage import io
import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import json
from static.torch_model.u2net import U2NETP  # small version u2net 4.7 MB
from static.torch_model.face import FaceNet, compare
"""
切换CPU与GPU 需注释7个位置
"""
# 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['png', 'jpg', 'jpeg']
# 保存文件地址
UPLOAD_FOLDER_RECOGNITION = './static/images/face_comparison'
# 新建蓝图
face_comparison_blueprint = Blueprint("face_comparison", __name__)
# 模型加载
net = U2NETP().cpu()
# net = U2NETP() # 切换GPU1
model_dir = 'static/torch_model/face_detect_model.pth'
net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
# net.load_state_dict(torch.load(model_dir)) # 切换GPU2
net.eval()

face_net = FaceNet().cpu()
# face_net = FaceNet() # 切换GPU3
face_modle_dir = 'static/torch_model/face_compare_model.pt'
face_net.load_state_dict(torch.load(face_modle_dir, map_location=torch.device('cpu')))
# face_net.load_state_dict(torch.load(face_modle_dir)) # 切换GPU4
face_net.eval()


def allowed_file(filename):
    """
    # 判断文件后缀是否在列表中
    :param filename:
    :return:
    """
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS


def contours_get(img0, img):
    img0 = img0.astype(np.uint8)

    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 3)

    gray = cv2.GaussianBlur(gray, (3, 3), 2)

    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    if binary.all() == None:
        print('hello')

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(binary))

    new_stats = stats[1:]

    if len(new_stats) >= 1:

        x, y, w, h, arae = new_stats[0]
        if arae < 10:
            x_min = 100
            y_min = 20
            x_max = 260
            y_max = 160
        else:
            x_min = x + 0.05 * w
            y_min = y + 0.05 * h
            x_max = x + w - 0.05 * w
            y_max = y + h - 0.1 * h
    else:
        x_min = 100
        y_min = 20
        x_max = 260
        y_max = 160
    img_cut = img[int(y_min):int(y_max), int(x_min):int(x_max)]

    return img_cut


def get_json(img_file, img_name, point_list, new_json_file):
    img_name = img_name.split('.j')[0]
    if len(point_list) == 0:
        a = {"path": img_file, "outputs": {}, "labeled": False}

    if len(point_list) > 0:
        a = {'path': img_file,
             'outputs': {'object': [{'name': '0',
                                     'bndbox': {'xmin': point_list[0], 'ymin': point_list[1], 'xmax': point_list[2],
                                                'ymax': point_list[3]}},
                                    ]},
             'labeled': True, 'size': {'width': 512, 'height': 512, 'depth': 3}}
    b = json.dumps(a)
    f2 = open(new_json_file + '\{0}.json'.format(img_name), 'w')
    f2.write(b)


def save_output(image_name, pred):
    predict = pred

    predict = predict.squeeze()

    predict_np = predict.cpu().data.numpy()
    predict_np = (predict_np + 0.5) * 255
    predict_np = np.transpose(predict_np, (1, 2, 0))

    im = Image.fromarray(predict_np * 255).convert('RGB')
    im.show()
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)
    img = cv2.cvtColor(np.asarray(pb_np), cv2.COLOR_RGB2BGR)

    return img


def behind_img(path):
    img0 = cv2.imread(path)

    if img0.shape[2] == 1:
        print('image is False')
    elif img0.all() == None:
        print('image is False')
    else:

        w = img0.shape[0]
        h = img0.shape[1]

        if w > h:

            new_img = cv2.copyMakeBorder(img0, 0, 0, (w - h) // 2, (w - h) // 2, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))

        else:
            new_img = cv2.copyMakeBorder(img0, (h - w) // 2, (h - w) // 2, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))

        img_cv = cv2.resize(new_img, (320, 320), interpolation=cv2.INTER_CUBIC)
        img_pil = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

        # img = (Image.open(path)).convert("RGB")
        # if img == None:
        #     print('image is False')

        return img_cv, img_pil


def img_resize(img0, size):
    w = img0.shape[0]
    h = img0.shape[1]

    if w > h:
        wer = w - h
        new_img = cv2.copyMakeBorder(img0, 0, 0, (w - h) // 2, (w - h) // 2, cv2.BORDER_CONSTANT,
                                     value=(0, 0, 0))
    else:
        new_img = cv2.copyMakeBorder(img0, (h - w) // 2, (h - w) // 2, 0, 0, cv2.BORDER_CONSTANT,
                                     value=(0, 0, 0))

    img_cv = cv2.resize(new_img, (size, size), interpolation=cv2.INTER_CUBIC)

    return img_cv


def tf(img, size1, size2):
    tf0 = transforms.Compose([
        transforms.Resize((size1, size2)),  # 形状压缩到416*416
        transforms.ToTensor(),  # 图像压缩到0~1之间，并转换为tensor
        # transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))
    ])

    img = tf0(img)
    img = torch.unsqueeze(img, 0)

    return img


def mask_read2value(img):
    # # 转换为灰度图
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray =gray*255
    gray = img[:, :, 0:1] * 255

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    binary = cv2.medianBlur(binary, 3)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(binary))  # 得到重心，长度等值

    sort_point = centroids[np.lexsort([centroids.T[0]])]

    return sort_point


def ten2cv(img, size):  #

    img0 = torch.cat((img[0][0], img[1][0], img[2][0]), 0)
    img1 = torch.cat((img[0][1], img[1][1], img[2][1]), 0)

    img0 = img0 * 255
    img1 = img1 * 255

    img0 = img0.cpu().data.numpy()
    img1 = img1.cpu().data.numpy()

    if img0.shape[0] == 3:
        img0 = np.transpose(img0, (1, 2, 0))
        img0 = cv2.resize(img0, (size, size))

        img1 = np.transpose(img1, (1, 2, 0))
        img1 = cv2.resize(img1, (size, size))
    else:
        img0 = None
        img1 = None

    return img0, img1


def read_img(img_dir):
    all_img_dir = []
    list = os.listdir(img_dir)
    for j in list:
        img_dir0 = os.path.join(img_dir, j)
        list0 = os.listdir(img_dir0)
        for i in list0:

            str = i.split('.')[1]
            dir_ = os.path.join(img_dir0, i)

            if str == 'mp4':
                k_list = os.listdir(dir_)
                for k in k_list:
                    img_ = os.path.join(dir_, k)
                    name = j + '\{0}.jpg'.format(np.random.randint(1, 2500000))
            else:
                img_ = dir_
                name = j + '\{0}.jpg'.format(np.random.randint(2500000, 5000000))
            all_img_dir.append([img_, name])

    return all_img_dir


@face_comparison_blueprint.route("/face_comparison", methods=['POST', 'GET'])
def face_comparsion_define():
    if request.method == 'POST':
        img_first = request.files['file_first']
        img_second = request.files['file_second']
        if img_first and allowed_file(img_first.filename):
            # secure_filename方法会去掉文件名中的中文
            try:
                img_first_name = secure_filename(img_first.filename)
                img_second_name = secure_filename(img_second.filename)

                img_first.save(os.path.join(UPLOAD_FOLDER_RECOGNITION, img_first_name))
                img_second.save(os.path.join(UPLOAD_FOLDER_RECOGNITION, img_second_name))

                img_path0 = os.path.join(UPLOAD_FOLDER_RECOGNITION, img_first_name)
                img_path1 = os.path.join(UPLOAD_FOLDER_RECOGNITION, img_second_name)

                cv_img0, pil_img0 = behind_img(img_path0)
                cv_img1, pil_img1 = behind_img(img_path1)
                tensor_img_0 = tf(pil_img0, 320, 320)
                tensor_img_1 = tf(pil_img1, 320, 320)
                img_cat = torch.cat((tensor_img_0, tensor_img_1), 0)
                img_cat = img_cat.cpu()

                # img_cat = img_cat.cuda() # 切换GPU5
                list_feture = net(img_cat)
                feture_img0, feture_img1 = ten2cv(list_feture, 320)
                cut_img0 = contours_get(feture_img0, cv_img0)
                cut_img0 = img_resize(cut_img0, 200)
                cut_img1 = contours_get(feture_img1, cv_img1)
                cut_img1 = img_resize(cut_img1, 200)
                # cv2.imwrite('d', cut_img1)
                tensor_cut0 = tf(Image.fromarray(cv2.cvtColor(cut_img0, cv2.COLOR_BGR2RGB)), 200, 200)
                tensor_cut1 = tf(Image.fromarray(cv2.cvtColor(cut_img1, cv2.COLOR_BGR2RGB)), 200, 200)

                # tensor_cut0 = tensor_cut0.cuda() # 切换GPU6
                tensor_cut0 = tensor_cut0.cpu()
                # tensor_cut1 = tensor_cut1.cuda() # 切换GPU7
                tensor_cut1 = tensor_cut1.cpu()
                feture_cut0 = face_net(tensor_cut0)
                feture_cut1 = face_net(tensor_cut1)
                sim = compare(feture_cut0, feture_cut1)
                sim_result = sim.cpu().data.numpy()[0][0]
                print(sim_result)
                return {"code": 200,
                        "message": '对比成功',
                        'result': {
                            'similarity': str(sim_result)}
                        }
            except(Exception):
                return{"code": 500,
                       "message": '对比失败',
                       "result": {}
                       }
        else:
            return {"code": 404,
                    "message": "格式错误，仅支持jpg、png、jpeg格式文件",
                    "result": {}
                    }
    # 上线时注释
    # else:
    #     return {"code": '503',
    #             "message": "仅支持post方法", "result": {}}
    return render_template('face_comparison.html')
