#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
#import matplotlib.pyplot as plt

def energy(img):  # take grayscale image
    dy, dx = np.gradient(img.astype(float))
    return np.abs(dx) + np.abs(dy)

def vseam_idx(color_img):  # take color image
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    r, c = img.shape
    V = np.zeros((r, c), dtype=np.int)  # seam index
    cut_img = img.copy()
    for k in range(c):  # at the beginning of each cycle, k seams HAVE been cut
        seam = find_seam(cut_img[:,:c-k])
        for i in range(r):
            cut_img[i, seam[i]:-1] = cut_img[i, seam[i]+1:]  # cut the seam, leaving last pixel unchanged
            real_j = -1
            for j in range(c):
                if V[i, j]: continue
                real_j += 1
                if real_j == seam[i]:
                    V[i, j] = k + 1  # cut the k+1 th seam
                    break
    return V

def shrink(img, V, alpha=0.5, debug=False):  # take color image
    r, c = V.shape
    k = int(c * alpha)  # number of seams to cut
    if debug:
        img_debug = img.copy()
        img_debug[V <= k] = (0, 0, 255)
        return img_debug
    else:
        S_img = np.zeros((r, c-k, 3), dtype=np.uint8)
        for i in range(r):
            S_img[i] = img[i][V[i]>k]
        return S_img

def expand(img, V, beta=0.5, debug=False):  # take color image
    r, c = V.shape
    k = int(c * beta)  # number of seams to expand
    L_img = np.zeros((r, c+k, 3), dtype=np.uint8)
    for i in range(r):
        real_j = 0
        for j in range(c):
            if V[i, j] > k:
                L_img[i, real_j] = img[i, j]
                real_j += 1
            else:
                # Must devide by 2 before summing them, otherwise will have uint8 overflow issue(>255)
                L_img[i, real_j] = (img[i, j-1] / 2.0 + img[i, j] / 2.0) if j-1 >= 0 else img[i, j]
                L_img[i, real_j+1] = (img[i, j] / 2.0 + img[i, j+1] / 2.0) if j+1 < c else img[i, j]
                if debug:
                    L_img[i, real_j] = (0, 0, 255)  # Show seams in red color in debug mode
                real_j += 2
    return L_img

def find_seam(img):  # take grayscale image
    r, c = img.shape
    if c == 1:
        return [0] * r
    M = np.ones((r, c)) * float('inf')  #  DP matrix of accumulated minimum energy, Dtype = float
    P = np.zeros((r, c), dtype=np.int)  # Store the previous j of each pixel in the seam path
    E = energy(img)
    M[0] = E[0]
    for i in range(1, r):
        pi = i - 1
        for j in range(c):
            for pj in (j-1, j, j+1):
                if pj < 0 or pj >= c: continue
                if M[pi, pj] < M[i, j]:
                    M[i, j] = M[pi, pj]
                    P[i, j] = pj
            M[i, j] += E[i, j]
    
    seam = [ np.argmin(M[r-1]) ]
    for i in reversed(range(1, r)):
        seam.append(P[i, seam[-1]])
                
    return seam[::-1]

def find_seam_fwd(img):  # take grayscale image
    r, c = img.shape
    M = np.ones((r, c)) * float('inf')  # DP matrix of accumulated minimum energy, Dtype = float
    P = np.zeros((r, c), dtype=np.int)  # Store the previous j of each pixel in the seam path
    #I = cv2.copyMakeBorder(img.astype(float), 1, 0, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)
    I = cv2.copyMakeBorder(img.astype(float), 1, 0, 1, 1, borderType=cv2.BORDER_REPLICATE)
    for i in range(r):
        ii = i + 1
        for j in range(c):
            jj = j + 1
            CU = np.abs(I[ii, jj+1] - I[ii, jj-1])
            CL = CU + np.abs(I[ii-1, jj] - I[ii, jj-1])
            CR = CU + np.abs(I[ii-1, jj] - I[ii, jj+1])
            ML = M[i-1, j-1] if i-1 >= 0 and j-1 >= 0 else 0
            MU = M[i-1, j] if i-1 >= 0 else 0
            MR = M[i-1, j+1] if i-1 >= 0 and j+1 < c else 0
            for fwd, pj in zip((ML+CL, MU+CU, MR+CR), (j-1, j, j+1)):
                if pj < 0 or pj >= c: continue
                if fwd < M[i, j]:
                    M[i, j] = fwd
                    P[i, j] = pj
    
    seam = [ np.argmin(M[r-1]) ]
    for i in reversed(range(1, r)):
        seam.append(P[i, seam[-1]])
                
    return seam[::-1]

def vseam_idx_fwd(color_img):  # take color image
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    r, c = img.shape
    V = np.zeros((r, c), dtype=np.int)  # seam index
    cut_img = img.copy()
    for k in range(c):  # at the beginning of each cycle, k seams HAVE been cut
        seam = find_seam_fwd(cut_img[:,:c-k])
        for i in range(r):
            cut_img[i, seam[i]:-1] = cut_img[i, seam[i]+1:]  # cut the seam, leaving last pixel unchanged
            real_j = -1
            for j in range(c):
                if V[i, j]: continue
                real_j += 1
                if real_j == seam[i]:
                    V[i, j] = k + 1  # cut the k+1 th seam
                    break
    return V

wtfall = cv2.imread('images/fig5.png')
V_wtfall = vseam_idx(wtfall)
wtfall1 = shrink(wtfall, V_wtfall, 0.5)
cv2.imwrite('fig5.png', wtfall1)  # Fig5, 2007

dolphin = cv2.imread('images/fig8.png')
V_dolphin = vseam_idx(dolphin)
dolphin1 = expand(dolphin, V_dolphin, 0.5)
V_dolphin1 = vseam_idx(dolphin1)
dolphin2 = expand(dolphin1, V_dolphin1, 0.34)
cv2.imwrite('fig8d_07.png', dolphin1)  # Fig8, part d, 2007
cv2.imwrite('fig8f_07.png', dolphin2)  # Fig8, part f, 2007

dolphin1_debug = expand(dolphin, V_dolphin, 0.5, debug=True)
cv2.imwrite('fig8c_07.png', dolphin1_debug)  # Fig8, part c, 2007

bench = cv2.imread('images/fig8-2008.png')
V_bench_bwd = vseam_idx(bench)
bench_bwd = shrink(bench, V_bench_bwd, 0.5)
cv2.imwrite('fig8Comp_backward_08.png', bench_bwd) # Fig8, bench BE, 2008

bench_bwd_seam = shrink(bench, V_bench_bwd, 0.5, debug=True)
cv2.imwrite('fig8Seam_backward_08.png', bench_bwd_seam)  # Fig8, bench BE seam, 2008

V_bench_fwd = vseam_idx_fwd(bench)
bench_fwd = shrink(bench, V_bench_fwd, 0.5)
cv2.imwrite('fig8Comp_forward_08.png', bench_fwd)  # Fig8, bench FE, 2008

bench_fwd_seam = shrink(bench, V_bench_fwd, 0.5, debug=True)
cv2.imwrite('fig8Seam_forward_08.png', bench_fwd_seam)  # Fig8, bench FE seam, 2008

car = cv2.imread('images/fig9-2008.png')
V_car_bwd = vseam_idx(car)
car_bwd = expand(car, V_car_bwd, 0.5)
cv2.imwrite('fig9Comp_backward_08.png', car_bwd)  # Fig9, car BE, 2008

V_car_fwd = vseam_idx_fwd(car)
car_fwd = expand(car, V_car_fwd, 0.5)
cv2.imwrite('fig9Comp_forward_08.png', car_fwd)  # Fig9, car FE, 2008
