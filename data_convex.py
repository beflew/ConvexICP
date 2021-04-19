#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

# Part of the code is referred from: https://github.com/charlesq34/pointnet

# # !ls
# # !pwd
BASE_DIR='c:/Users/yshao/Downloads/VOCtrainval_11-May-2012/DL/ICP/ConvexICP-master/ConvexICP-master'


def download():
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


# +
# download()
# -

def load_data(partition):
    download()
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


# Pick one
item=5
def load_data_single(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data[item], all_label[item]


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud



# Convexity conditions
# load an picture
# generate 3 point clouds






# What is factor
factor=1
def RoT_random(pointcloud,lam=1,wt=None):
    l_w=[]
        
    if (wt != None):
#         w=wt
        for e in wt:
            l_w.append(lam*e)
        
    else:
        w=lam*np.random.normal(-0.5,0.5);l_w.append(w)
        
        tw=0.1
        w_t1=lam*np.random.uniform(-tw,tw);l_w.append(w_t1)
        w_t2=lam*np.random.uniform(-tw,tw);l_w.append(w_t2)
        w_t3=lam*np.random.uniform(-tw,tw);l_w.append(w_t3)
        
    # Rotation
    w = l_w[0]        
    # Translation
    l_translation=l_w[1:]
#     print('weights',wt)
        
        
    anglex = w * np.pi / factor
    angley = w * np.pi / factor
    anglez = w * np.pi / factor

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                    [0, cosx, -sinx],
                    [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                    [0, 1, 0],
                    [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])
    R_ab = Rx.dot(Ry).dot(Rz)
    R_ba = R_ab.T
    
    
    translation_ab = np.array(l_translation)
    translation_ba = -R_ba.dot(translation_ab)
    
    pointcloud1 = pointcloud.T

    rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
    pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)    
    
    
    # R_ab,translateion_ab - Wi
    return pointcloud2,l_w,(rotation_ab,translation_ab,R_ab)

# +
# # !pip install --upgrade matplotlib
# # !conda install matplotlib --force
# -



# +
class ModelNet40Convex(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, unseen=False, factor=4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        
        # Which Object
#         self.item=10
        
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        # which ever number, it is a fixed item
#         item=self.item
        pointcloud = self.data[item][:self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        
        # perturbed cloud1
        pointcloud1,wt1,mat_w1=RoT_random(pointcloud)
        #permutation invariant - pointcloud
        pointcloud1=np.random.permutation(pointcloud1.T).T
        
        # perturbed cloud2
        pointcloud2,wt2,mat_w2=RoT_random(pointcloud)
        #permutation invariant - pointcloud
        pointcloud2=np.random.permutation(pointcloud2.T).T
        
        # convexity condition cloud3
        lam=0.8
        lam1=lam
        lam2=1-lam
    #         w3=lam*w1+(1-lam)*w2
        # apply perturbation twice with scalar coefficeint
        pointcloud3_0, _ , mat_w3_0 =RoT_random(pointcloud,lam=lam1,wt=wt1)
        
        pointcloud3, _ , mat_w3_1=RoT_random(pointcloud3_0.T,lam=lam2,wt=wt2)
        
        pointcloud3=np.random.permutation(pointcloud3.T).T
        
        return pointcloud.astype('float32'),pointcloud1.astype('float32'),pointcloud2.astype('float32'),pointcloud3.astype('float32'),\
            mat_w1[1:],mat_w2[1:],mat_w3_0[1:],mat_w3_1[1:]
    
#     R_ab.astype('float32'), \
#                translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
#                euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


# -

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, unseen=False, factor=4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


# +
# # if __name__ == '__main__':
# train = ModelNet40(1024)
# test = ModelNet40(1024, 'test')
# for data in train:
#     print(len(data))
#     break
# -

train_cvx=ModelNet40Convex(1024)

# +
# from torch.optim.lr_scheduler import MultiStepLR
# # from data import ModelNet40,download,load_data

# # from data import ModelNet40Convex,download,load_data
# from data_convex import ModelNet40Convex,download,load_data
# import numpy as np
# from torch.utils.data import DataLoader

# # out_cvx
# train_cvx_loader = DataLoader(train_cvx, batch_size=128, shuffle=True, drop_last=True)

# +
# for batch_idx, content in enumerate(train_cvx_loader):
#     (src, pointcloud1, pointcloud2, pointcloud3,mat1,mat2,mat3_0,mat3_1)= content
#     print(batch_idx)
#     break

# +
# len(mat1)

# +
# translation_ab1,R_ab1=mat1[0],mat1[1]

# translation_ab2,R_ab2=mat2[0],mat2[1]

# translation_ab3_0,R_ab3_0=mat3_0[0],mat3_0[1]
# translation_ab3_1,R_ab3_1=mat3_1[0],mat3_1[1]

# translation_ab3=translation_ab3_0+translation_ab3_1
# R_ab3=R_ab3_0.matmul(R_ab3_1)

# +
# a=R_ab3[0,:,:]
# a0=R_ab3_0[0,:,:]
# a1=R_ab3_1[0,:,:]

# +
# a,a0.matmul(a1)

# +
# anglex = np.random.uniform() * np.pi / 1
# angley = np.random.uniform() * np.pi / 1
# anglez = np.random.uniform() * np.pi / 1

# cosx = np.cos(anglex)
# cosy = np.cos(angley)
# cosz = np.cos(anglez)
# sinx = np.sin(anglex)
# siny = np.sin(angley)
# sinz = np.sin(anglez)
# Rx = np.array([[1, 0, 0],
#                 [0, cosx, -sinx],
#                 [0, sinx, cosx]])
# Ry = np.array([[cosy, 0, siny],
#                 [0, 1, 0],
#                 [-siny, 0, cosy]])
# Rz = np.array([[cosz, -sinz, 0],
#                 [sinz, cosz, 0],
#                 [0, 0, 1]])
# R_ab = Rx.dot(Ry).dot(Rz)

# +
# anglex,angley,anglez

# +
# np.random.uniform(0,1)

# +
# mat=Rx.dot(Ry).dot(Rz)
# mat.dot(mat)
# -


