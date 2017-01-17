# encoding: utf-8

"""
@author: liubo
@software: PyCharm
@file: tmp.py
@time: 2017/1/11 17:30
@contact: ustb_liubo@qq.com
@annotation: tmp
"""
import sys
import logging
from logging.config import fileConfig
import os
import cv2

reload(sys)
sys.setdefaultencoding("utf-8")
# fileConfig('logger_config.ini')
# logger_error = logging.getLogger('errorhandler')


img_path = '360build.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (400, 400))
cv2.imwrite('360build_small.jpg', img)
if __name__ == '__main__':
    pass
