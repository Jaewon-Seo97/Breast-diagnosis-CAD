import os, glob
import cv2
import SimpleITK as sitk
import numpy as np
import utils.prep_utils as prep



class _get_image():
    def __init__(self, dcm_path, reSize, r_erode=None):
        self.dcm_path = dcm_path
        self.r_erode = r_erode
        self.reSize = reSize

    def ImgFromDcm(self, name, saveimg=False):    #######

        dcmfile = sitk.ReadImage(os.path.join(self.dcm_path, 'dcm', f'{name}.dcm'))
        dcm_arry = sitk.GetArrayFromImage(dcmfile)[0]

        WL = dcmfile.GetMetaData('0028|1050')
        WW = dcmfile.GetMetaData('0028|1051')

        WL = int(WL.split('\\')[0])
        WW = int(WW.split('\\')[0])
        
        dcm_default8 = prep.win_scale(dcm_arry, WL, WW, np.uint8, (0,(2**8)-1))

        return dcm_default8, dcm_arry, dcmfile

    def get_BRegion(self, dcm_default8):

        # dcm_default8 = self.ImgFromDcm(name)
        remain = dcm_default8.copy()
        remain[dcm_default8 <=dcm_default8.min()] = 0
        remain[dcm_default8>dcm_default8.min()] = 255

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
        

        if self.r_erode!=0:
            kernelE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.r_erode,self.r_erode))
            erode_region = cv2.erode(origin_breast_region,kernelE)       

            b, binary_erode = cv2.threshold(erode_region, 0, 255, cv2.THRESH_BINARY) 
            removed_boundary = origin_breast_region.copy()
            removed_boundary[binary_erode==0] = 0
            removed_boundary[binary_erode!=0] = 255
        else:
            removed_boundary = None
            pass

        return origin_breast_region, removed_boundary, [min(cont_xlocs), min(cont_ylocs), max(cont_xlocs), max(cont_ylocs)]

    def get_inputdata(self, name, img_raw, dcm_default8, wGT=True):
        kind = name.split('_')[-1]
        img_region, _, BRlocList=  self.get_BRegion(dcm_default8)

        
        get_scaling = prep._sorted_scaling(img_raw[BRlocList[0]:BRlocList[2],BRlocList[1]:BRlocList[3]], img_region)
        multi_img = get_scaling.apply_scale()

        data_x = prep.resize_centerpadding(imageSize=self.reSize, in_arr=multi_img, dtype = np.uint8, color = True)

        if wGT:
            msk = cv2.imread(os.path.join(self.dcm_path, 'msk', f'{name}.png'),0)
            mask_region = msk[BRlocList[0]:BRlocList[2],BRlocList[1]:BRlocList[3]]
            data_y = prep.resize_centerpadding(imageSize=self.reSize, in_arr=mask_region, dtype = np.uint8, color = False)        

            data_l = mk_maskLabel(prep.category_mask(data_y, img_region), kind)

            return data_x, data_y, data_l, BRlocList
        else:
            return data_x, BRlocList


def _dcm_save(img, dcmfile, path_save, name):
    src_dcm = np.concatenate((np.expand_dims(img, axis= 0),np.expand_dims(img, axis= 0)), axis=0)
                            
    tar_dcm = sitk.GetImageFromArray(src_dcm)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()   
    for k in dcmfile.GetMetaDataKeys():
        tar_dcm.SetMetaData(k, dcmfile.GetMetaData(k))

    writer.SetFileName(os.path.join(path_save, 'dcm',f'{name}.dcm'))
    writer.Execute(tar_dcm[:,:,0]) 

    return 0

def mk_maskLabel(msk, kind):
    label = np.zeros((1,3), np.float)
    if msk[:,:,0].max() == 0:
        label[0,2] = 1.0
    else:
        if kind == 'benign':
            label[0,0] = 1.0 
        else:
            label[0,1] = 1.0   
    return label