import numpy as np
import cv2
from PIL import Image
import torch
import math
from torchvision import transforms


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
        if w <= h:
            new_img = cv2.copyMakeBorder(img0, (h - w) // 2, (h - w) // 2, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))

        img_cv = cv2.resize(new_img, (416, 416), interpolation=cv2.INTER_CUBIC)
        img_pil = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

        # img = (Image.open(path)).convert("RGB")
        # if img == None:
        #     print('image is False')

        return img_cv, img_pil


def tf(img, size1, size2):
    tf = transforms.Compose([
        transforms.Resize((size1, size2)),  # 形状压缩到416*416
        transforms.ToTensor(),  # 图像压缩到0~1之间，并转换为tensor
        # transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))
    ])

    img = tf(img)
    img = torch.unsqueeze(img, 0)

    return img


def ten2cv(img):  #

    img = torch.cat((img[0], img[1], img[2]), 1)[0]
    img = img.cpu().data.numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    else:
        img = img
        img = np.concatenate((img, img, img), 1)
        img = img[0]
        img = np.transpose(img, (1, 2, 0))

    return img


def mask_read2value(img):
    # # 转换为灰度图
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray =gray*255
    gray = img[:, :, 0:1] * 255

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    binary = cv2.medianBlur(binary, 3)
    # binary=cv2.GaussianBlur(binary,(3,3),1.3)

    # corners=cv2.cornerHarris(binary, 8,5,0.05)#水平和垂直方向的阈值，步长，迭代精度

    # 锐化，边缘变得更加清晰
    # corners=cv2.dilate(corners,None)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(binary))  # 得到重心，长度等值

    sort_point = centroids[np.lexsort([centroids.T[0]])]

    if 6 <= len(sort_point):
        sort_point = np.delete(sort_point, 6, axis=0)
    books = r''
    for i in sort_point:
        number = int(i[1] // 10)

        if number == 10:
            k = ' '
        elif number == 11:
            k = '.'
        else:
            k = number

        books += str(k)

    return books


def mask2value(img):  # 此处输入为3通道cv图片，若输入特征图，需要转类型
    if img.shape[2] == 3:
        list = []
        img = img * 255
        for i in range(8):
            gray = img[:, :, 2:3]
            thre = 30 * i + 10
            t1 = (gray < thre + 1)
            t2 = (thre - 5 <= gray)
            gray = t1 & t2
            gray = np.array(gray + 0, dtype=np.uint8)
            # print(type(gray))
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # print(type(contours))
            if len(contours) == 0:
                continue
            else:
                def cnt_area(cnt):
                    area = cv2.contourArea(cnt)
                    return area

                contours.sort(key=cnt_area, reverse=False)

                rect = cv2.minAreaRect(contours[-1])

                # 得到最小矩形的坐标
                box = cv2.boxPoints(rect)
                # 标准化坐标到整数
                box = np.int0(box)
                # 画出边界
                # cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

                list.append([i, box])

    elif img.shape[2] == None or img.shape[2] == 1:

        list = []
        for i in range(8):
            gray = img[:, :, :]
            # print(gray.shape)
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thre = 30 * i + 10
            t1 = (gray < thre + 1)
            t2 = (thre - 5 <= gray)
            gray = t1 & t2
            gray = np.array(gray + 0, dtype=np.uint8)
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                continue
            else:
                def cnt_area(cnt):
                    area = cv2.contourArea(cnt)
                    return area

                print(type(contours))
                contours.sort(key=cnt_area, reverse=False)

                rect = cv2.minAreaRect(contours[-1])
                # 得到最小矩形的坐标
                box = cv2.boxPoints(rect)
                # 标准化坐标到整数
                box = np.int0(box)
                # 画出边界
                # cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
                list.append([i, box])

    return list  # 返回矩形框类型和四个坐标组成的box


def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def get_cut_img1(img, boxs):
    if len(boxs) < 2:
        ap = img[208:218, 208:218]
        end_img0 = ap
        end_img2 = ap
        end_img7 = ap
        img_x = ap

    else:
        len_x0 = math.pow(((boxs[0][1][1][0] - boxs[0][1][0][0]) * (boxs[0][1][1][0] - boxs[0][1][0][0])
                           + (boxs[0][1][1][1] - boxs[0][1][0][1]) * (boxs[0][1][1][1] - boxs[0][1][0][1])),
                          0.5)  # 0框的长

        len_x_test = math.pow(((boxs[0][1][3][0] - boxs[0][1][2][0]) * (boxs[0][1][3][0] - boxs[0][1][2][0])
                               + (boxs[0][1][3][1] - boxs[0][1][2][1]) * (boxs[0][1][3][1] - boxs[0][1][2][1])),
                              0.5)
        # print('长度对比',len_x_test,len_x0)
        len_y0 = math.pow(((boxs[0][1][2][0] - boxs[0][1][1][0]) * (boxs[0][1][2][0] - boxs[0][1][1][0])
                           + (boxs[0][1][2][1] - boxs[0][1][1][1]) * (boxs[0][1][2][1] - boxs[0][1][1][1])),
                          0.5)  # 0框的宽
        len_x1 = math.pow(((boxs[1][1][1][0] - boxs[1][1][0][0]) * (boxs[1][1][1][0] - boxs[1][1][0][0])
                           + (boxs[1][1][1][1] - boxs[1][1][0][1]) * (boxs[1][1][1][1] - boxs[1][1][0][1])),
                          0.5)  # 1框的长
        len_y1 = math.pow(((boxs[1][1][2][0] - boxs[1][1][1][0]) * (boxs[1][1][2][0] - boxs[1][1][1][0])
                           + (boxs[1][1][2][1] - boxs[1][1][1][1]) * (boxs[1][1][2][1] - boxs[1][1][1][1])),
                          0.5)  # 1框的宽

        if len_x0 > len_y0:
            line1 = [boxs[0][1][0][0], boxs[0][1][0][1], boxs[0][1][1][0], boxs[0][1][1][1]]
            line10 = [boxs[0][1][2][0], boxs[0][1][2][1], boxs[0][1][1][0], boxs[0][1][1][1]]
            if len_x1 > len_y1:
                line2 = [boxs[1][1][1][0], boxs[1][1][1][1], boxs[1][1][2][0], boxs[1][1][2][1]]
                line20 = [boxs[1][1][3][0], boxs[1][1][3][1], boxs[1][1][2][0], boxs[1][1][2][1]]
            else:
                line2 = [boxs[1][1][1][0], boxs[1][1][1][1], boxs[1][1][0][0], boxs[1][1][0][1]]
                line20 = [boxs[1][1][3][0], boxs[1][1][3][1], boxs[1][1][0][0], boxs[1][1][0][1]]
        else:
            line1 = [boxs[0][1][2][0], boxs[0][1][2][1], boxs[0][1][1][0], boxs[0][1][1][1]]
            line10 = [boxs[0][1][0][0], boxs[0][1][0][1], boxs[0][1][1][0], boxs[0][1][1][1]]

            if len_x1 > len_y1:
                line2 = [boxs[1][1][1][0], boxs[1][1][1][1], boxs[1][1][2][0], boxs[1][1][2][1]]
                line20 = [boxs[1][1][3][0], boxs[1][1][3][1], boxs[1][1][2][0], boxs[1][1][2][1]]
            else:
                line2 = [boxs[1][1][1][0], boxs[1][1][1][1], boxs[1][1][0][0], boxs[1][1][0][1]]
                line20 = [boxs[1][1][3][0], boxs[1][1][3][1], boxs[1][1][0][0], boxs[1][1][0][1]]

        rout_point_x, rout_point_y = cross_point(line1, line2)
        rout_point_x0, rout_point_y0 = cross_point(line10, line20)

        new_angle_line0 = [boxs[0][1][1][0] - rout_point_x0, boxs[0][1][1][1] - rout_point_y0]
        new_angle_line = [rout_point_x - boxs[1][1][1][0], rout_point_y - boxs[1][1][1][1]]  # 方向向量

        line_angle0 = math.atan2(new_angle_line0[1], new_angle_line0[0])
        line_angle = math.atan2(new_angle_line[1], new_angle_line[0])
        line_angle0 = line_angle0 * 180 / math.pi
        line_angle = line_angle * 180 / math.pi

        print('角度值', line_angle0, line_angle)
        img0 = img
        img2 = img
        img7 = img

        routline = -line_angle - 90
        routline0 = -line_angle0 - 90

        if routline0 - 360 > 0:
            routline0 = routline

        ap = img[208:218, 208:218]
        end_img0 = ap
        end_img2 = ap
        end_img7 = ap

        for i in range(len(boxs)):

            if boxs[i][0] == 0:
                min_x = min(boxs[0][1][0][0], boxs[0][1][1][0], boxs[0][1][2][0], boxs[0][1][3][0])
                max_x = max(boxs[0][1][0][0], boxs[0][1][1][0], boxs[0][1][2][0], boxs[0][1][3][0])
                min_y = min(boxs[0][1][0][1], boxs[0][1][1][1], boxs[0][1][2][1], boxs[0][1][3][1])
                max_y = max(boxs[0][1][0][1], boxs[0][1][1][1], boxs[0][1][2][1], boxs[0][1][3][1])

                len_x = math.pow(((boxs[0][1][1][0] - boxs[0][1][0][0]) * (boxs[0][1][1][0] - boxs[0][1][0][0])
                                  + (boxs[0][1][1][1] - boxs[0][1][0][1]) * (boxs[0][1][1][1] - boxs[0][1][0][1])),
                                 0.5)  # 框的长
                len_y = math.pow(((boxs[0][1][2][0] - boxs[0][1][1][0]) * (boxs[0][1][2][0] - boxs[0][1][1][0])
                                  + (boxs[0][1][2][1] - boxs[0][1][1][1]) * (boxs[0][1][2][1] - boxs[0][1][1][1])),
                                 0.5)  # 框的宽
                cut_img0 = img0[min_y:max_y, min_x:max_x]
                x1 = int(180 / 2 - cut_img0.shape[0] / 2)
                x2 = int(180 / 2 + cut_img0.shape[0] / 2)
                y1 = int(180 / 2 - cut_img0.shape[1] / 2)
                y2 = int(180 / 2 + cut_img0.shape[1] / 2)
                mask0 = np.zeros([180, 180, 3], dtype=np.uint8)
                mask0 = mask0 + 0
                mask0[x1:x2, y1:y2] = cut_img0
                # center_point =[int(min_x/2+max_x/2),int(min_y/2+max_y/2)]
                # center_point=[cut_img.shape[0],cut_img.shape[1]]
                center_point = [90, 90]
                # center_point=[0,0]
                M = cv2.getRotationMatrix2D(center_point, routline, 1)
                img_new0 = cv2.warpAffine(mask0, M, (180, 180))  # 得到旋转后图像，中点在几何中心
                # print(routline)

                x11 = int(90 - len_x / 2)
                x22 = int(90 + len_x / 2)
                y11 = int(90 - len_y / 2)
                y22 = int(90 + len_y / 2)

                if x22 - x11 > y22 - y11:
                    end_img0 = img_new0[y11 - 1:y22 + 1, x11:x22]
                elif x22 - x11 < y22 - y11:
                    end_img0 = img_new0[x11 - 1:x22 + 1, y11:y22]

                # cv2.imshow('*-*-*-*-*-',end_img0)
                # cv2.waitKey(0)
            elif boxs[i][0] == 2:
                min_x = min(boxs[2][1][0][0], boxs[2][1][1][0], boxs[2][1][2][0], boxs[2][1][3][0])
                max_x = max(boxs[2][1][0][0], boxs[2][1][1][0], boxs[2][1][2][0], boxs[2][1][3][0])
                min_y = min(boxs[2][1][0][1], boxs[2][1][1][1], boxs[2][1][2][1], boxs[2][1][3][1])
                max_y = max(boxs[2][1][0][1], boxs[2][1][1][1], boxs[2][1][2][1], boxs[2][1][3][1])

                len_x = math.pow(((boxs[2][1][1][0] - boxs[2][1][0][0]) * (boxs[2][1][1][0] - boxs[2][1][0][0])
                                  + (boxs[2][1][1][1] - boxs[2][1][0][1]) * (boxs[2][1][1][1] - boxs[2][1][0][1])),
                                 0.5)  # 框的长
                len_y = math.pow(((boxs[2][1][2][0] - boxs[2][1][1][0]) * (boxs[2][1][2][0] - boxs[2][1][1][0])
                                  + (boxs[2][1][2][1] - boxs[2][1][1][1]) * (boxs[2][1][2][1] - boxs[2][1][1][1])),
                                 0.5)  # 框的宽

                cut_img2 = img2[min_y:max_y, min_x:max_x]

                x1 = int(180 / 2 - cut_img2.shape[0] / 2)
                x2 = int(180 / 2 + cut_img2.shape[0] / 2)
                y1 = int(180 / 2 - cut_img2.shape[1] / 2)
                y2 = int(180 / 2 + cut_img2.shape[1] / 2)
                mask2 = np.zeros([180, 180, 3], dtype=np.uint8)
                mask2 = mask2 + 0
                mask2[x1:x2, y1:y2] = cut_img2
                # center_point =[int(min_x/2+max_x/2),int(min_y/2+max_y/2)]
                # center_point=[cut_img.shape[0],cut_img.shape[1]]
                center_point = [90, 90]
                # center_point=[0,0]
                M = cv2.getRotationMatrix2D(center_point, routline, 1)
                img_new2 = cv2.warpAffine(mask2, M, (180, 180))  # 得到旋转后图像，中点在几何中心
                # print(routline)
                x10 = int(90 - len_x / 2)
                x20 = int(90 + len_x / 2)
                y10 = int(90 - len_y / 2)
                y20 = int(90 + len_y / 2)

                if x20 - x10 > y20 - y10:
                    end_img2 = img_new2[y10 - 1:y20 + 1, x10:x20]
                elif x20 - x10 < y20 - y10:
                    end_img2 = img_new2[x10 - 1:x20 + 1, y10:y20]

                # cv2.imshow('*-*-*-*-*-', img_new2)
                # cv2.imshow('*****', end_img2)
                # cv2.waitKey(0)

            elif boxs[i][0] == 7:

                min_x = min(boxs[7][1][0][0], boxs[7][1][1][0], boxs[7][1][2][0], boxs[7][1][3][0])
                max_x = max(boxs[7][1][0][0], boxs[7][1][1][0], boxs[7][1][2][0], boxs[7][1][3][0])
                min_y = min(boxs[7][1][0][1], boxs[7][1][1][1], boxs[7][1][2][1], boxs[7][1][3][1])
                max_y = max(boxs[7][1][0][1], boxs[7][1][1][1], boxs[7][1][2][1], boxs[7][1][3][1])
                len_x = math.pow(((boxs[7][1][1][0] - boxs[7][1][0][0]) * (boxs[7][1][1][0] - boxs[7][1][0][0])
                                  + (boxs[7][1][1][1] - boxs[7][1][0][1]) * (boxs[7][1][1][1] - boxs[7][1][0][1])),
                                 0.5)  # 框的长
                len_y = math.pow(((boxs[7][1][2][0] - boxs[7][1][1][0]) * (boxs[7][1][2][0] - boxs[7][1][1][0])
                                  + (boxs[7][1][2][1] - boxs[7][1][1][1]) * (boxs[7][1][2][1] - boxs[7][1][1][1])),
                                 0.5)  # 框的宽
                cut_img = img7[min_y:max_y, min_x:max_x]
                x1 = int(180 / 2 - cut_img.shape[0] / 2)
                x2 = int(180 / 2 + cut_img.shape[0] / 2)
                y1 = int(180 / 2 - cut_img.shape[1] / 2)
                y2 = int(180 / 2 + cut_img.shape[1] / 2)
                mask1 = np.zeros([180, 180, 3], dtype=np.uint8)
                mask1 = mask1 + 0
                mask = mask1
                mask[x1:x2, y1:y2] = cut_img
                # center_point =[int(min_x/2+max_x/2),int(min_y/2+max_y/2)]
                # center_point=[cut_img.shape[0],cut_img.shape[1]]
                center_point = [90, 90]
                # center_point=[0,0]
                M = cv2.getRotationMatrix2D(center_point, routline, 1)
                img_new7 = cv2.warpAffine(mask, M, (180, 180))  # 得到旋转后图像，中点在几何中心
                # print(routline)
                x17 = int(90 - len_x / 2)
                x27 = int(90 + len_x / 2)
                y17 = int(90 - len_y / 2)
                y27 = int(90 + len_y / 2)

                if x27 - x17 > y27 - y17:
                    end_img7 = img_new7[y17 - 1:y27 + 1, x17:x27]
                elif x27 - x17 < y27 - y17:
                    end_img7 = img_new7[x17 - 1:x27 + 1, y17:y27]

            else:
                img_x = ap

    return end_img0, end_img2, end_img7, img_x


def img_trans(img, size):
    tr = transforms.Compose([
        transforms.Resize(size),
    ])

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transforms.ToTensor()(img)
    img0 = img

    for i in range(1, img.shape[2] // img.shape[1]):
        img = torch.cat((img0, img), 1)

    img = tr(img)
    img = torch.unsqueeze(img, 0)  # nchw

    return img


if __name__ == '__main__':
    pass
    # img_file=r'D:\project_vienat\data\test\ocr_test\4.jpg'
    # img_file2=r'D:\project_vienat\OCR\train_data\202106051611066553514427.jpeg'
    # img=cv2.imread(img_file)
    # img2=cv2.imread(img_file2)
    # boxs=mask2value(img) #得到外接矩形boxs
    #
    # img_0,img_2,img_7,img_x=get_cut_img(img2,boxs) #得到想得到的数字切片
    #
    # img_0=img_trans(img_0,(120,120))
    # img_2=img_trans(img_2,(120,120))
    # img_7 =img_trans(img_7,(120, 120))
    # print(img_2.shape)
    #
    # for c in boxs:#可视化
    #     name = c[0]
    #     box = np.int0(c[1])
    #     # 画出边界
    #     cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

    # a=np.array([0,3,1,5,9,7])
    # b=(a<8)
    # c=(a>3)
    #
    # print(b&c)
