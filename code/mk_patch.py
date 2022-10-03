import os, glob
import cv2
import SimpleITK as sitk
import numpy as np
import math
from roifile import ImagejRoi



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



class _get_image():
    def __init__(self, dcm_path, r_erode, reSize):
        self.dcm_path = dcm_path
        self.r_erode = r_erode
        self.reSize = reSize

    def ImgFromDcm(self, name, saveimg=False):

        dcmfile = sitk.ReadImage(os.path.join(self.dcm_path, 'dcm', f'{name}.dcm'))
        dcm_arry = sitk.GetArrayFromImage(dcmfile)[0]

        WL = dcmfile.GetMetaData('0028|1050')
        WW = dcmfile.GetMetaData('0028|1051')

        WL = int(WL.split('\\')[0])
        WW = int(WW.split('\\')[0])
        
        dcm_default8 = win_scale(dcm_arry, WL, WW, np.uint8, (0,(2**8)-1))

        return dcm_default8, dcmfile

    def get_BRegion(self, dcm_default8):

        # dcm_default8 = self.ImgFromDcm(name)
        remain = dcm_default8.copy()
        remain[dcm_default8 <=0 ] = 0
        remain[dcm_default8>0] = 255

        ret, thr = cv2.threshold(remain, 127, 255, 0)
        try:
            _, contours, _ = cv2.findContours(thr, cv2.RETR_TREE, 2)
        except:
            contours, _ = cv2.findContours(thr, cv2.RETR_TREE, 2)

        areas=[]
        for cnt in contours:
            area= cv2.contourArea(cnt)
            areas.append(area)

        cont_xlocs =[]
        cont_ylocs =[]
        for chp in range(len(contours[areas.index(max(areas))])):
            cont_xlocs.append(contours[areas.index(max(areas))][chp][0][1])
            cont_ylocs.append(contours[areas.index(max(areas))][chp][0][0])

        crop = np.zeros(remain.shape, dtype= np.uint8)
        poly = cv2.fillPoly(crop, [contours[areas.index(max(areas))]], (255,255,255))
        dcm_default8[poly == 0] = 0
        origin_breast_region = dcm_default8[min(cont_xlocs):max(cont_xlocs),min(cont_ylocs):max(cont_ylocs)]

        kernelE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.r_erode,self.r_erode))
        erode_region = cv2.erode(origin_breast_region,kernelE)       

        b, binary_erode = cv2.threshold(erode_region, 0, 255, cv2.THRESH_BINARY) 
        removed_boundary = origin_breast_region.copy()
        removed_boundary[binary_erode==0] = 0
        removed_boundary[binary_erode!=0] = 255

        return origin_breast_region, removed_boundary, [min(cont_xlocs), min(cont_ylocs), max(cont_xlocs), max(cont_ylocs)]

    def get_rois(self, originshape, name):
        
        oneRois = sorted(glob.glob(os.path.join(self.dcm_path, 'roi', f'{name}_*.roi')))
        gt_boxes = np.ndarray((len(oneRois),4), dtype = np.uint16)
        ori_RegionMask = np.zeros(originshape, dtype= np.uint8)
        if len(oneRois) != 0:
            for g, oner in enumerate(oneRois):

                one_roi = ImagejRoi.fromfile(oner)
                ori_loc= one_roi.left, one_roi.top, one_roi.right, one_roi.bottom
                ori_RegionMask[ori_loc[1]:ori_loc[3],ori_loc[0]:ori_loc[2]]= 255
                
                for i in range(4):
    
                    gt_boxes[g][i] = ori_loc[i]
                    
        else:
            ori_loc= None, None, None, None
            pass

        # io.imsave(os.path.join(self.dcm_path, 'nii', f'{name}.png'), ori_RegionMask, check_contrast=False)
        return ori_RegionMask, gt_boxes

    def get_PatchArray(self,partial_region):
        partial_region = cv2.resize(partial_region, dsize=(self.reSize,self.reSize), interpolation=cv2.INTER_LINEAR)
        partial_region = np.expand_dims(cv2.cvtColor(partial_region, cv2.COLOR_GRAY2RGB), axis=0)

        return partial_region

    def get_patches(self, img, name, reSize):
        
        imagenames = []
        x_array = np.ndarray((0, reSize, reSize, 3), np.uint8)
        origin_breast_region, removed_boundary, BRlocList=  self.get_BRegion(img)
        _, gt_boxes = self.get_rois(img.shape, name)

        n_xpatch = math.ceil(origin_breast_region.shape[1]/reSize)
        n_ypatch = math.ceil(origin_breast_region.shape[0]/reSize)

        x_strides = int(origin_breast_region.shape[1]/n_xpatch)
        y_strides = int(origin_breast_region.shape[0]/n_ypatch)

        
        for xt in range(0,origin_breast_region.shape[1],x_strides):
            for yt in range(0,origin_breast_region.shape[0],y_strides):
                

                partial_region = origin_breast_region[yt:yt+reSize,xt:xt+reSize]
                check_black = removed_boundary[yt:yt+reSize,xt:xt+reSize]

                if check_black.sum()<=(reSize*reSize*255)/2:  
                    pass

                else:
                    imagenames.append(f'{name}_{xt+BRlocList[1]}_{yt+BRlocList[0]}')
                    
                    partial_region = cv2.resize(partial_region, dsize=(reSize,reSize), interpolation=cv2.INTER_LINEAR)
                    color_partial = np.expand_dims(cv2.cvtColor(partial_region, cv2.COLOR_GRAY2RGB), axis=0)
                    x_array = np.concatenate((x_array, color_partial), axis=0)

        return x_array, imagenames, gt_boxes

def _dcm_save(img, dcmfile, path_save, name):
    src_dcm = np.concatenate((np.expand_dims(img, axis= 0),np.expand_dims(img, axis= 0)), axis=0)
              
    tar_dcm = sitk.GetImageFromArray(src_dcm)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()   
    excludeList= ["0028|0002", "0028|0100", "0028|0101",  "0028|0102", "0028|0103", "ITK_original_direction", "ITK_original_spacing"]
    for k in dcmfile.GetMetaDataKeys():
        if k not in excludeList:
            try:
                tar_dcm.SetMetaData(k, dcmfile.GetMetaData(k))
            except:
                 pass
            
    writer.SetFileName(os.path.join(path_save, f'{name}.dcm'))
    writer.Execute(tar_dcm[:,:,0]) 

    return tar_dcm