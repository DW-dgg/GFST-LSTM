import csv
import glob
import os

import numpy as np
from imageio import imread
from torch.utils.data import Dataset
import torch

up_path = {
    70: "Radar",
    35: "Wind",
    10: "Precip"
}
low_path = {
    70: "radar_",
    35: "wind_",
    10: "precip_"
}


class Radar(Dataset):
    def __init__(self, root, is_train, factor, input_length, transform=None):
        super(Radar, self).__init__()
        assert factor in [70, 35, 10], "factor {} dose not legal.".format(factor)
        self.root = root
        self.input_length = input_length
        self.factor = factor
        self.n_path = up_path[factor]
        self.n_low_path = low_path[factor]
        if is_train:
            self.images_path = 'Train'
            path_list0 = self.load_csv(filename="dataset_train.csv", root=root)
            self.path_list = path_list0
        else:
            self.path_list = self.load_csv(filename="dataset_testA.csv", root=root)
            self.images_path = 'TestA'

    def __getitem__(self, idx):
        images = []
        images_path = self.path_list[idx][:40]####20gai30
        for path in images_path:
            Tc_image = self.image_read(path)
            images.append(Tc_image)
        images = torch.stack(images, dim=0)
        return images

    def __len__(self):
        return len(self.path_list)

   # def padding_img(self, data):
   #     padding_data = np.zeros((128, 128))
   #    padding_data[13:-14, 13:-14] = data
   #     return padding_data

    def load_csv(self, filename, root):
        path = os.path.join(root, filename)
        if (not os.path.exists(path)):
            raise IOError("{} does not exist".format(filename))
        img = []
        with open(path, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img.append(row)
        ## 返回的长度是[2825,41]
        return img

    def image_read(self, sinle_image):
        image_name = self.n_low_path+sinle_image
        image_path = os.path.join(self.root, self.images_path, self.n_path, image_name)
        image = np.array(imread(image_path) / 255.0, np.float64)
        
        t_image = torch.from_numpy(image).float()
        tc_image = torch.unsqueeze(t_image, dim=0)

        return tc_image


# class RadarTestB1(Dataset):
#     def __init__(self, root, factor, input_length, directory="TestB1/"):
#         super(RadarTestB1, self).__init__()
#         assert factor in [70, 35, 10], "factor {} dose not legal.".format(factor)
#         self.root = root
#         self.input_length = input_length
#         self.factor = factor
#         self.category = root + directory + up_path[factor]
#         self.path_list = self.load_csv(root=root, factor=factor,
#                                        filename='TestB1.csv')
#
#     def __getitem__(self, idx):
#         images = []
#         dirname = self.path_list[idx][0]
#         images_path = self.path_list[idx][20 - self.input_length + 1:]
#         for path in images_path:
#             Tc_image = self.image_read(path)
#             images.append(Tc_image)
#         images = torch.stack(images, dim=0)
#         return images, dirname
#
#     def __len__(self):
#         return len(self.path_list)
#
#     def image_read(self, sinle_image):
#         image = np.array(imread(sinle_image) / 255.0, dtype=np.float64)
#         t_image = torch.from_numpy(image).float()
#         tc_image = torch.unsqueeze(t_image, dim=0)
#         return tc_image
#
#     def load_csv(self, root, factor=70, filename='TestB1.csv'):
#         filename = up_path[factor] + filename
#         if not os.path.exists(os.path.join(root, filename)):
#             dirs = os.listdir(self.category)
#             with open(os.path.join(root, filename), mode='w', newline='') as f:
#                 writer = csv.writer(f)
#                 for dir in dirs:
#                     images = []
#                     images += glob.glob(os.path.join(self.category, dir, '*.png'))
#                     writer.writerow([dir] + images)
#         images = []
#         with open(os.path.join(self.root, filename)) as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 images.append(row)
#
#         return images

if __name__ == '__main__':
    data = Radar(root='../../tianchi_data', is_train=False, factor=70, input_length=10)
    a = data.__getitem__(0)
    print(a.shape)
