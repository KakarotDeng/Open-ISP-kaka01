import cv2
import numpy as np
import imageio
import rawpy
from PIL import Image

#1. 读取tiff文件。用dcraw获取的未经任何处理的raw图
raw_origin = cv2.imread('raw_1.tiff',-1)
# ratio_shrink = 1
# raw_shrink = cv2.resize(raw_origin, (int(raw_origin.shape[1] / ratio_shrink), int(raw_origin.shape[0]/ ratio_shrink)))
raw_shrink = raw_origin
print(raw_origin.shape)
print(raw_shrink.shape)
cv2.namedWindow('step0',cv2.WINDOW_KEEPRATIO)
cv2.imshow('step0', raw_origin)
cv2.namedWindow('step0.5',cv2.WINDOW_KEEPRATIO)
cv2.imshow('step0.5', raw_shrink)
# cv2.waitKey(0)

#2. 非线性校正
black_level = 0
saturation_level = 16383
raw_linear = (raw_shrink - black_level)/(saturation_level - black_level)
cv2.namedWindow('step1',cv2.WINDOW_KEEPRATIO)
cv2.imshow('step1', raw_linear)


#3. 对颜色通道进行比例放缩，小白平衡
WB_multipliers = [1.875000, 1.000000, 1.816406]
bayer_filter = 'rggb'
origin_multipler = np.ones([raw_linear.shape[0], raw_linear.shape[1]])*WB_multipliers[1]

def fliter2multiper(orgin_mul, mul_para, bayer_type):
    if bayer_type == 'bggr':
        orgin_mul[0::2, 0::2] = mul_para[2]
        orgin_mul[1::2, 1::2] = mul_para[0]
    elif bayer_type == 'rggb':
        orgin_mul[0::2, 0::2] = mul_para[0]
        orgin_mul[1::2, 1::2] = mul_para[2]
    elif bayer_type == 'grbg':
        orgin_mul[0::2, 1::2] = mul_para[0]
        orgin_mul[1::2, 0::2] = mul_para[2]
    elif bayer_type == 'gbrg':
        orgin_mul[0::2, 1::2] = mul_para[2]
        orgin_mul[1::2, 0::2] = mul_para[0]

    return orgin_mul

processed_multiper = fliter2multiper(origin_multipler, WB_multipliers, bayer_filter)
raw_first_bw = raw_linear*processed_multiper
cv2.namedWindow('step2',cv2.WINDOW_KEEPRATIO)
cv2.imshow('step2', raw_first_bw)


#4. 色彩插值（去马赛克）
def Bilinear_Demosaic(input_raw, bayer_type):
    R_blanks = np.zeros([input_raw.shape[0], input_raw.shape[1]])
    G_blanks = np.zeros([input_raw.shape[0], input_raw.shape[1]])
    B_blanks = np.zeros([input_raw.shape[0], input_raw.shape[1]])
    RGB_blanks = np.zeros([input_raw.shape[0], input_raw.shape[1], 3])

    if bayer_type == 'bggr':
        R_blanks[1::2, 1::2] = 1
        G_blanks[1::2, 0::2] = 1
        G_blanks[0::2, 1::2] = 1
        B_blanks[0::2, 0::2] = 1
    elif bayer_type == 'rggb':
        R_blanks[0::2, 0::2] = 1
        G_blanks[1::2, 0::2] = 1
        G_blanks[0::2, 1::2] = 1
        B_blanks[1::2, 1::2] = 1
    elif bayer_type == 'grbg':
        R_blanks[0::2, 1::2] = 1
        G_blanks[0::2, 0::2] = 1
        G_blanks[1::2, 1::2] = 1
        B_blanks[1::2, 0::2] = 1
    elif bayer_type == 'gbrg':
        R_blanks[1::2, 0::2] = 1
        G_blanks[0::2, 0::2] = 1
        G_blanks[1::2, 1::2] = 1
        B_blanks[0::2, 1::2] = 1

    def outsize_mat(input_mat):
        output_mat = np.zeros([input_mat.shape[0] + 2, input_mat.shape[1] + 2])
        output_mat[1:output_mat.shape[0] - 1, 1:output_mat.shape[1] - 1] = input_mat
        return output_mat

    new_mat = outsize_mat(input_raw)
    Rblk_max = outsize_mat(R_blanks)

    def RB_shape_the_blk(RGB_blk, input, i, j, newmap, sign):
        RGB_blk[i][j][sign] = input[i][j]
        l1 = input.shape[0] - 1
        l2 = input.shape[1] - 1
        K = 4
        K2 = 4
        if i == 0 or i == l1 or j == 0 or j == l2:
            K = 3
            K2 = 2
            if abs(i - 0.5 * l1) + abs(j - 0.5 * l2) == 0.5 * (l1 + l2):
                K = 2
                K2 = 1

        RGB_blk[i][j][1] = (newmap[i][j + 1] + newmap[i + 2][j + 1] + newmap[i + 1][j] + newmap[i + 1][j + 2]) / K
        RGB_blk[i][j][2 - sign] = (newmap[i][j] + newmap[i + 2][j] + newmap[i][j + 2] + newmap[i + 2][j + 2]) / K2
        return RGB_blk

    def G_shape_the_blk(RGB_blk, input, i, j, newmap, Rblk_max):
        RGB_blk[i][j][1] = input[i][j]
        l1 = input.shape[0] - 1
        l2 = input.shape[1] - 1
        if i == 0 or i == l1 or j == 0 or j == l2:
            if i == 0 or i == l1:
                if Rblk_max[i + 2][j + 1] + Rblk_max[i][j + 1] == 0:
                    RGB_blk[i][j][2] = newmap[i + 2][j + 1] + newmap[i][j + 1]
                    RGB_blk[i][j][0] = (newmap[i + 1][j + 2] + newmap[i + 1][j]) * 0.5
                else:
                    RGB_blk[i][j][0] = newmap[i + 2][j + 1] + newmap[i][j + 1]
                    RGB_blk[i][j][2] = (newmap[i + 1][j + 2] + newmap[i + 1][j]) * 0.5
            if j == 0 or j == l2:
                if Rblk_max[i + 2][j + 1] + Rblk_max[i][j + 1] == 0:
                    RGB_blk[i][j][2] = (newmap[i + 2][j + 1] + newmap[i][j + 1]) * 0.5
                    RGB_blk[i][j][0] = newmap[i + 1][j + 2] + newmap[i + 1][j]
                else:
                    RGB_blk[i][j][0] = (newmap[i + 2][j + 1] + newmap[i][j + 1]) * 0.5
                    RGB_blk[i][j][2] = newmap[i + 1][j + 2] + newmap[i + 1][j]
            if abs(i - 0.5 * l1) + abs(j - 0.5 * l2) == 0.5 * (l1 + l2):
                if Rblk_max[i + 2][j + 1] + Rblk_max[i][j + 1] == 0:
                    RGB_blk[i][j][2] = newmap[i + 2][j + 1] + newmap[i][j + 1]
                    RGB_blk[i][j][0] = newmap[i + 1][j + 2] + newmap[i + 1][j]
                else:
                    RGB_blk[i][j][0] = newmap[i + 2][j + 1] + newmap[i][j + 1]
                    RGB_blk[i][j][2] = newmap[i + 1][j + 2] + newmap[i + 1][j]
        else:
            if Rblk_max[i + 2][j + 1] + Rblk_max[i][j + 1] == 0:
                RGB_blk[i][j][2] = (newmap[i + 2][j + 1] + newmap[i][j + 1]) * 0.5
                RGB_blk[i][j][0] = (newmap[i + 1][j + 2] + newmap[i + 1][j]) * 0.5
            else:
                RGB_blk[i][j][0] = (newmap[i + 2][j + 1] + newmap[i][j + 1]) * 0.5
                RGB_blk[i][j][2] = (newmap[i + 1][j + 2] + newmap[i + 1][j]) * 0.5
        return RGB_blk

    for i in range(input_raw.shape[0]):
        for j in range(input_raw.shape[1]):
            if R_blanks[i][j] == 1:
                RGB_blanks = RB_shape_the_blk(RGB_blanks, input_raw, i, j, new_mat, 0)
            if G_blanks[i][j] == 1:
                RGB_blanks = G_shape_the_blk(RGB_blanks, input_raw, i, j, new_mat, Rblk_max)
            if B_blanks[i][j] == 1:
                RGB_blanks = RB_shape_the_blk(RGB_blanks, input_raw, i, j, new_mat, 2)

    return RGB_blanks


raw_demosaic = Bilinear_Demosaic(raw_first_bw, 'rggb')

raw_demosaic1 = raw_demosaic *255
raw_demosaic2 = raw_demosaic1.astype('uint8')
cv2.namedWindow('step3',cv2.WINDOW_KEEPRATIO)
cv2.imshow('step3', raw_demosaic)
# cv2.waitKey(0)

#5. 大白平衡
def WhiteBalance(img, method):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    if method == 'grayworld':
        Gray = (np.average(R)+np.average(G)+np.average(B))/3
        kr = Gray/np.average(R)
        kg = Gray / np.average(G)
        kb = Gray / np.average(B)
        R *= kr
        G *= kg
        B *= kb
        img[:, :, 0] = R
        img[:, :, 1] = G
        img[:, :, 2] = B
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(3):
                    if img[i][j][k]>=255:
                        img[i][j][k]=255
        return img

    if method == 'perfectreflection':
        Rgb_map = R+G+B
        map_one = Rgb_map.reshape([Rgb_map.shape[0]*Rgb_map.shape[1]])
        length=map_one.shape[0]
        map_two = sorted(map_one)
        ratio = 0.02
        ratio_value = np.array(((1-ratio)*length)).astype('int')
        standard = map_two[ratio_value]
        r0,g0,b0,num=0,0,0,0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j][0]+img[i][j][1]+img[i][j][2]:
                    r0+=img[i][j][0]
                    g0+=img[i][j][1]
                    b0+=img[i][j][2]
                    num+=1
        rT = r0/num
        gT = g0/num
        bT = b0/num
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                r1 = img[i][j][0]*(255/rT)
                g1 = img[i][j][1] * (255 / gT)
                b1 = img[i][j][2] * (255 / bT)
                if r1>=255:
                    r1=255
                if g1>=255:
                    g1=255
                if b1>=255:
                    b1=255
                img[i][j][0]=r1
                img[i][j][1]=g1
                img[i][j][2]=b1
        k=1
        return img


img_afterwb = WhiteBalance(raw_demosaic1, 'grayworld')
img_afterwb = img_afterwb.astype('uint8')
cv2.namedWindow('step4',cv2.WINDOW_KEEPRATIO)
cv2.imshow('step4', img_afterwb)
cv2.waitKey(0)



