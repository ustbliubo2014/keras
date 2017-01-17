# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: datagan.py
@time: 2017/1/11 12:09
@contact: ustb_liubo@qq.com
@annotation: datagen : 数据扩充
"""
import sys
import logging
from logging.config import fileConfig
import os
from keras.preprocessing.image import ImageDataGenerator
import cv2
import pdb
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


if __name__ == '__main__':

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,   # randomly flip images
        shear_range=0.3,   # 图片拉伸比例(一般小于0.4)
        zoom_range=0.0,    # 焦距变化
    )

    test_str = 'shear_range'
    src_folder = 'D:\data\Anthony'
    dst_folder = 'D:\data\Anthony_augment/{}'.format(test_str)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    src_pic_list = os.listdir(src_folder)
    data = []
    for pic in src_pic_list[:1]:
        image_path = os.path.join(src_folder, pic)
        img = cv2.imread(image_path)
        data.append(img)
    data = np.transpose(np.asarray(data, dtype=np.float32), (0, 3, 1, 2))
    datagen.fit(data)
    index = 0
    for img_k in datagen.flow(data, batch_size=1):
        img_k = np.transpose(img_k, (0, 2, 3, 1))
        cv2.imwrite('{}/{}_{}.jpg'.format(dst_folder, test_str, index), img_k[0])
        index += 1
        if index > 20:
            break
