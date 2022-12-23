#!/usr/bin/env python
# coding: utf-8

# Refer to:
# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

def compare_images(imageA, imageB):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, multichannel=True)
    return m, s

my_result = ['fig5.png', 'fig8d_07.png', 'fig8f_07.png', 'fig8Comp_backward_08.png', 'fig8Comp_forward_08.png',
             'fig9Comp_backward_08.png', 'fig9Comp_forward_08.png' ]
author_result = ['author_waterfallNarrow.png', 'author_dolphinStretch1.png', 'author_dolphinStretch2.png',
                'author_bench_bwd.png', 'author_bench_fwd.png', 'author_car_expand_bwd.png', 'author_car_expand_fwd.png']

my_path = './'
author_path = 'images/Author_results/'
print("%-30s%10s%10s" % ('Replica', 'MSE', 'SSIM'))
print('-'*50)
for my_img, author_img in zip(my_result, author_result):
    m, s = compare_images(cv2.imread(my_path + my_img), cv2.imread(author_path + author_img))
    print("%-30s%10.2f%10.2f" % (my_img, m, s))
