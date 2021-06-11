
# import os
# import torch
# from torch.utils.data import Dataset
# from torchvision.transforms.functional import to_tensor
# from PIL import Image
# from scipy.signal import convolve2d
# import numpy as np
# import h5py
# import random
# import model.common as common
# from option import args
#
# def default_loader(path):
#     return Image.open(path).convert('L')  # RGB-->Gray
#
#
# def LocalNormalization(patch, P=3, Q=3, C=1):
#     kernel = np.ones((P, Q)) / (P * Q)
#     patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
#     patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
#     patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
#     patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
#     return patch_ln
#
#
# def NonOverlappingCropPatches(im, patch_size=32, stride=32):
#     w, h = im.size
#     patches = ()
#     for i in range(0, h - stride, stride):
#         for j in range(0, w - stride, stride):
#             patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
#             patch = LocalNormalization(patch[0].numpy())
#             patches = patches + (patch,)  # great !!!
#     return patches
#
# def NonOverlappingCropPatches_random(im, gt, patch_size=32, stride=32):
#     w, h = im.size # 8000, 4000
#     rnd_h = random.randint(0, max(0, h - patch_size))
#     rnd_w = random.randint(0, max(0, w - patch_size))
#
#     im_crop = im.crop((rnd_w, rnd_h, rnd_w + patch_size, rnd_h + patch_size))
#     im_crop = np.asarray(im_crop)
#
#     gt_crop = gt.crop((rnd_w, rnd_h, rnd_w + patch_size, rnd_h + patch_size))
#     gt_crop = np.asarray(gt_crop)
#
#
#
#     #im_crop = torch.from_numpy()
#     return im_crop, gt_crop
#
#
# class IQADataset(Dataset):
#     def __init__(self, conf, exp_id=0, status='train', loader=default_loader):
#         self.imp_num = conf['imp_num']
#         self.loader = loader
#         self.imrefID = conf['yl360Dataset']['refimpID_pth']
#         self.im_dir = conf['yl360Dataset']['img_ref_IMG_pth']
#         self.patch_size = conf['patch_size']
#         self.stride = conf['stride']
#         self.bz = conf['batch_size']
#         self.dwt = common.DWT()
#         self.indexData = np.arange(self.imp_num)
#         # np.random.seed(1000)
#         # random.seed(1000)
#         np.random.shuffle(self.indexData)
#         if os.path.exists(os.path.join(args.log_dir_MW, "train_test_randList.txt")):
#             self.indexData = np.loadtxt(os.path.join(args.log_dir_MW, "train_test_randList.txt"))
#         else:
#             np.random.shuffle(self.indexData)
#             np.savetxt(os.path.join(args.log_dir_MW, "train_test_randList.txt"), self.indexData)
#         test_ratio = conf['test_ratio']
#         train_ratio = conf['train_ratio']
#         trainindex = self.indexData[: int(train_ratio*self.imp_num)]
#         valindex = self.indexData[int(train_ratio*self.imp_num): int((1 - test_ratio) * self.imp_num)+1]
#         testindex = self.indexData[int((1 - test_ratio) * self.imp_num): ]
#
#         if status == 'train':
#             self.index = trainindex
#             print(len(self.index))
#             print("# Train Images: {}".format(len(self.index)))
#             print('Ref Index:')
#             print(trainindex)
#         if status == 'test':
#             self.index = testindex
#             print(len(self.index))
#             print("# Test Images:  {}".format(len(self.index)))
#             print('Test Index:')
#             print(testindex)
#         if status == 'val':
#             self.index = valindex
#             print(len(self.index))
#             print("# Val Images: {}".format(len(self.index)))
#
#
#     def __len__(self):
#         return int(len(self.index)) # 960 / 5 =192
#
#     def __getitem__(self, idx):
#         imp_id = self.index[idx]
#         #print(imp_id)
#         #print(np.loadtxt(self.imrefID, dtype=str).shape)
#         gt_name = np.loadtxt(self.imrefID, dtype=str)[int(imp_id), 0]
#         imp_name = np.loadtxt(self.imrefID, dtype=str)[int(imp_id), 1]
#         #print(gt_name)
#         gt = self.loader('.'.join([os.path.join(self.im_dir, gt_name), 'jpg']))
#         imp = self.loader('.'.join([os.path.join(self.im_dir, imp_name), 'jpg']))
#         #gt_crop_np = NonOverlappingCropPatches_random(gt, self.patch_size, self.stride)
#         imp_crop_np, gt_crop_np = NonOverlappingCropPatches_random(imp, gt, self.patch_size, self.stride)
#
#         gt_crop_tor = torch.from_numpy(gt_crop_np).unsqueeze(0).unsqueeze(1)
#         imp_crop_tor = torch.from_numpy(imp_crop_np).unsqueeze(0).unsqueeze(1)
#
#
#         gt_crop_dwt = self.dwt(gt_crop_tor).squeeze(0)
#         imp_crop_dwt = self.dwt(imp_crop_tor).squeeze(0)
#
#         gt_crop_iwt = torch.from_numpy(gt_crop_np).unsqueeze(0)
#         imp_corp_iwt = torch.from_numpy(imp_crop_np).unsqueeze(0)
#
#         return imp_crop_dwt.float().cuda()/255, gt_crop_dwt.float().cuda()/255, imp_corp_iwt.float().cuda()/255, gt_crop_iwt.float().cuda()/255
#         # 返回一个img的一个patch以及其标签。


# Implemented by Li Yang
# Email: 13021041@buaa.edu.cn
# Date: 2021/6/6

import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import h5py
import random
#import model.common as common
#from option import args

def default_loader(path):
    return np.asarray(Image.open(path))  # RGB-->Gray


class IQADataset(Dataset):
    def __init__(self, conf, status='train', loader=default_loader):
        np.random.seed(1000)
        random.seed(1000)

        self.train_img = conf.train_img_dir
        self.test_img = conf.test_img_dir
        self.train_gt = conf.train_gt_dir
        self.test_gt = conf.test_gt_dir
        self.train_id = conf.train_id
        self.test_id = conf.test_id

        train_num = len(os.listdir(self.train_img))
        test_num = len(os.listdir(self.test_img))


        self.loader = loader

        trainindex = np.arange(train_num)

        testindex = np.arange(test_num)

        if status == 'train':
            self.index = trainindex
            self.id = self.train_id
            print(len(self.index))
            print("# Train Images: {}".format(len(self.index)))

        if status == 'test':
            self.index = testindex
            print(len(self.index))
            self.id = self.test_id
            print("# Test Images:  {}".format(len(self.index)))


    def __len__(self):
        return int(len(self.index)) #

    def __getitem__(self, idx):
        imp_id = self.index[idx]
        #print(imp_id)
        #print(np.loadtxt(self.imrefID, dtype=str).shape)

        img_name_split = np.loadtxt(self.id, dtype=str)[int(imp_id)]
        #print('The training image is {}'.format(img_name_split))
        gt_name = img_name_split + '.txt'
        imp_name = img_name_split + '.jpg'
        #print(gt_name)
        gt = np.loadtxt(os.path.join(self.train_gt, gt_name))
        imp = self.loader(os.path.join(self.train_img , imp_name))
        #gt_crop_np = NonOverlappingCropPatches_random(gt, self.patch_size, self.stride)

        dec_input_tensor = torch.from_numpy(gt[:, 0:2])
        gt_tensor = torch.from_numpy(gt[:, 2:4])
        img_tensor = torch.from_numpy(imp).permute(2,0,1)



        return img_tensor.float().cuda()/255, dec_input_tensor.float().cuda(), gt_tensor.float().cuda()
        # 返回一个img的一个patch以及其标签。
