import os

import h5py
from PIL import Image
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as F
from tqdm import tqdm

def get_vision_stats():
    to_tensor = transforms.ToTensor()
    max_h, max_w = 0, 0
    images = []
    i = 0
    accu_mean = []
    accu_std = []

    for filename in os.listdir('/datashare/train2014/'):
        i += 1
        img = Image.open('/datashare/train2014/'+filename).convert('RGB')
        img = F.resize(img, (640, 640))
        img_t = to_tensor(img)

        h, w = img.height, img.width
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w

        images.append(img_t)

        if i % 1000 == 0:
            images = np.ndarray(images)
            images_mean = np.mean(images, axis=(0, 2, 3)) / 255
            images_std = np.std(images, axis=(0, 2, 3)) / 255
            accu_mean.append(images_mean)
            accu_std.append(images_std)
            images = []

    images_mean = np.array(accu_mean).mean(axis=0)
    images_std = np.array(accu_std).mean(axis=0)

    print(f'images mean is: {images_mean}, images std is: {images_std}')
    print(f'max height is: {max_h}, max weidth is: {max_w}')


def create_vision_files(cfg):
    transforms = create_transforms(cfg)
    for data_name in ['train', 'val']:

        fpath = cfg['vision_utils'][f'{data_name}_file_path']
        h5_file = h5py.File(fpath, mode='a')
        N = cfg['vision_utils'][f'num_{data_name}_imgs']  # num of images in dataset
        h_size = cfg['dataset']['resize_h']
        w_size = cfg['dataset']['resize_w']
        imgs = h5_file.create_dataset('imgs', shape=(N, 3, h_size, w_size), dtype=np.float16, fillvalue=0)
        ids = h5_file.create_dataset('img_ids', shape=(N,), dtype='int32')
        for i, filename in tqdm(enumerate(os.listdir(f'/datashare/{data_name}2014/'))):
            img = Image.open(f'/datashare/{data_name}2014/' + filename).convert('RGB')
            img = transforms(img)
            img_id = filename.split("_")[-1].split('.')[0].lstrip('0') #COCO_train2014_0000imageid.jpg
            imgs[i, :, :, :] = img.numpy().astype('float16')
            ids[i] = int(img_id)
            if i % 10000 == 0:
                print(f'finished uploading {i*10000} items')

        h5_file.close()
        print(f'done creating {data_name} vision file')


def create_transforms(cfg):
    resize_h, resize_w, resize_int = cfg['dataset']['resize_h'], cfg['dataset']['resize_w'], cfg['dataset']['resize_int']
    resize_size = resize_int if resize_int > 0 else (resize_h, resize_w)

    transform = transforms.Compose([
        transforms.Resize(size=resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229, 0.224, 0.225])])
    return transform

def get_num_of_imgs():
    for data in ['train', 'val']:
        dir = f'/datashare/{data}2014/'
        list = os.listdir(dir)  # dir is your directory path
        number_files = len(list)
        print(f'num of images in dataset: {data} is {number_files}')