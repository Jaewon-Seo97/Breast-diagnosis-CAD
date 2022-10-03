import os, glob
from skimage import io
import cv2
import SimpleITK as sitk
import numpy as np


def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """
    
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1]-1)
    
    data_new[data <= (wl-ww/2.0)] = out_range[0]
    
    data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
    
    data_new[data > (wl+ww/2.0)] = out_range[1]-1
    
    return data_new.astype(dtype)

def apply_clahe(breast_region):
    clahe = cv2.createCLAHE(clipLimit =0.005, tileGridSize = (8,8))
    clahe_img= clahe.apply(breast_region.copy())
    clahe_img[breast_region == 0] = 0

    return clahe_img

def mk_multi_img(imageList):
    multi_img = np.ndarray(imageList[0].shape +(len(imageList),), dtype=np.uint8)
    for m in range(len(imageList)):
        multi_img[:,:,m] = imageList[m]
        
    return multi_img

def resize_centerpadding(imageSize, in_arr, dtype = np.uint8, color = True):
    img_stand = max(in_arr.shape)
    scale = imageSize/img_stand
    short_length = round((min(in_arr.shape[0], in_arr.shape[1])*scale))
    imgnpy = np.zeros((imageSize,imageSize)+(in_arr.shape[-1],), dtype=dtype) if color else np.zeros((imageSize,imageSize), dtype=dtype)

    if img_stand == in_arr.shape[0]:
        
        resized = cv2.resize(in_arr, dsize=(short_length,imageSize), interpolation=cv2.INTER_AREA)
        if color:
            imgnpy[:imageSize, imageSize//2-short_length//2:imageSize//2+(short_length-short_length//2),:] = resized[:,:,:]
        else:
            imgnpy[:imageSize, imageSize//2-short_length//2:imageSize//2+(short_length-short_length//2)] = resized[:,:]
    else:
        resized= cv2.resize(in_arr, dsize=(imageSize,short_length), interpolation=cv2.INTER_AREA)
        if color:
            imgnpy[imageSize//2-short_length//2:imageSize//2+(short_length-short_length//2),:imageSize,:] = resized[:,:,:]
        else:
            imgnpy[imageSize//2-short_length//2:imageSize//2+(short_length-short_length//2),:imageSize] = resized[:,:]

    return imgnpy

def category_mask(mass_mask, img, mask_channel):

        # mass_mask = mask_binary(mass_mask)
    BR_mask = np.zeros(mass_mask.shape, dtype = np.uint8)
    BR_mask[img!=0] = 255
    # BR_mask[mass_mask!=0] = 0
    back_mask = np.ndarray(mass_mask.shape, dtype = np.uint8)
    back_mask[mass_mask!=0] = 0
    back_mask[BR_mask==0] = 255

    BR_mask = cv2.bitwise_xor(BR_mask, mass_mask)
    if BR_mask.shape != mass_mask.shape:
        BR_mask = np.expand_dims(BR_mask, axis= -1)

    mask_category = np.concatenate((mass_mask, BR_mask, back_mask), axis=-1)

    return mask_category



class _sorted_scaling():
    def __init__(self, img_raw, img_region8):
        self.img_raw = img_raw
        self.img_region8 = img_region8
        self.img_raw[img_region8==0] = 0
        self.none_zero = img_raw[img_raw>0]

    def sorted_quarter(self):

        to_max = max(self.none_zero).astype(np.float16)
        to_min = min(self.none_zero).astype(np.float16)
        WW = (to_max - to_min)*3/4.0
        WL = (to_max + to_min)/2.0
        
        return WW, WL

    def sorted_minmax(self):
        
        to_max = max(self.none_zero).astype(np.float16)
        to_min = min(self.none_zero).astype(np.float16)
        WW = (to_max - to_min)
        WL = (to_max + to_min)/2.0
               
        return WW, WL

    def apply_scale(self):

        WW, WL = self.sorted_minmax()
        img_minmax = apply_clahe(win_scale(data = self.img_raw, wl=WL, ww=WW, dtype=np.uint8, out_range=(0,255)))
        WW, WL = self.sorted_quarter()
        img_quarter = win_scale(data = self.img_raw, wl=WL, ww=WW, dtype=np.uint8, out_range=(0,255))

        multiList= [self.img_region8, img_minmax, img_quarter] if (WW < WL) else [img_quarter, img_minmax, self.img_region8]

        multi_img = mk_multi_img(multiList)

        return multi_img







