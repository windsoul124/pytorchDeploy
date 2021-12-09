from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

dictionary = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
              '17', '18', '19', '20', '21',
              '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38',
              '39', '40', '41', '42', '43'
                                      '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56',
              '57', '58', '59', '60', '61', '62', '63', '64', '65', '66',
              '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83',
              '84', '85', '86''87', '88', '89'
    , '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106',
              '107', '108', '109', '110',
              '111', '112']

# sos=['82','83','84','85','86','87','88','89'
#             ,'90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110',
#             '111','112','113','114','115','116','117','118','119','120']
tf = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    # transforms.Normalize(std=(0.5,0.5,0.5),mean=(0.5,0.5,0.5))
])


# def img_crop(img):
#     w,h=img.size
#     if w>=h:
#         img=img.crop((0,0,w,w))
#     if w<h:
#         img=img.crop((0,0,h,h))
#
#     return img

class MyDataset(Dataset):

    def __init__(self, main_dir):

        self.dataset = []
        for face_dir in os.listdir(main_dir):
            for face_filename in os.listdir(os.path.join(main_dir, face_dir)):
                self.dataset.append([os.path.join(main_dir, face_dir, face_filename), int(face_dir)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        # img = cv2.imread(data[0])
        # img = Image.fromarray(img.astype('uint8')[:, :, ::-1], mode='RGB')
        # img=Image.open(data[0]).convert("RGB")
        img = (Image.open(data[0])).convert("RGB")
        img_data = tf(img)
        return img_data, data[1]


if __name__ == '__main__':
    mydataset = MyDataset(r"")
    print()
