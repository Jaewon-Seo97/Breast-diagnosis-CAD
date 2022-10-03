
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import pandas as pd
import cv2
from mk_patch import _get_image, _dcm_save
import numpy as np
import time

from keras.models import load_model
from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout
import argparse

class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape(inputs)[i] if sh is None else sh for i, sh in enumerate(self.noise_shape)])

customObjects = {
    'swish': nn.swish,
    'FixedDropout': FixedDropout
}


def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def count_indicators(GT_boxes, predict_boxes, th_iou=0.1):
    # getting confusion matrix indicators
    global cal_iou
    TPs, FPs, FNs = 0, 0, 0

    # for normal case (no mass)
    if len(GT_boxes) ==0 :
        FPs += len(predict_boxes)

    # counting mass
    else:
        for ri in range(len(predict_boxes)):
            fp_count = False
            tp_count = 0
            for gi in range(len(GT_boxes)):

                cal_iou = IoU(GT_boxes[gi].astype(np.float32), predict_boxes[ri].astype(np.float32))
                if cal_iou > th_iou:
                    tp_count += 1
                else:
                    fp_count = True

            if fp_count == True and tp_count == 0:
                FPs+=1
            else:
                TPs +=tp_count
                
        FNs = len(GT_boxes) - TPs

    return TPs, FPs, FNs

def get_performance(all_TPs, all_FPs, all_FNs, n_img, alpha = 0.0001):
    sensitivity= all_TPs/(all_TPs+all_FNs+alpha)
    prec = all_TPs/(all_TPs+all_FPs+alpha)
    fppi = all_FPs/n_img 

    return sensitivity, prec, fppi

def trun_n_d(n,d):
    return (n if not n.find('.')+1 else n[:n.find('.')+d+1])  

def check_imges(img, Pd_boxes, GT_boxes, abprob, wGT = False, fsclae= 3, font = cv2.FONT_HERSHEY_COMPLEX_SMALL):
    
    kind_text = f'Abnormality score: {trun_n_d(str(abprob*100),1)}% '
    fsize, BaseLine=cv2.getTextSize(kind_text,font,fsclae,2)
    korg = ((img.shape[1]//2)-fsize[0]//2, (img.shape[0] - 50))
    img = cv2.putText(img, kind_text, korg, font, 
                    fontScale=fsclae, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

    for ori_loc in Pd_boxes:
        cv2.rectangle(img, (ori_loc[0], ori_loc[1]), (ori_loc[2], ori_loc[3]), (255,0,0),5)

    if wGT == False:
        return [img]
    else:
        F_GT = img.copy()
        for ori_loc in GT_boxes:
            cv2.rectangle(img, (ori_loc[0], ori_loc[1]), (ori_loc[2], ori_loc[3]),(0,255,0),5)  
        return [F_GT, img]

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser()
    parser.add_argument('--gpus',                   help='gpu number', default='0')
    parser.add_argument('--wGT',      action = 'store_true' )
    return parser.parse_args(args)

def main(args=None):
    
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpus)

    r_erode = 100
    d_radius = 12
    reSize = 224
    prob_th = 0.6

    path_init = os.path.abspath('..')
    get_image = _get_image(dcm_path = os.path.join(path_init, 'data'), r_erode=r_erode, reSize = reSize)

    #########################
    # get nameList
    #########################
    test_names = [name.split('.')[0] for name in os.listdir(os.path.join(path_init, 'data', 'dcm')) if name.endswith('.dcm')]

    ################
    # loading model
    ################
    print(f'\n# Loading model...')
    modelpath = os.path.join(path_init, 'model')
    model = load_model(os.path.join(modelpath, f'model_MGCal.h5'), custom_objects=customObjects)
    print(f'\tModel load success')

    ###################
    # saving results
    ###################

    path_result = os.path.join(path_init, f'results')
    path_saveImg= [os.path.join(path_result,'img'), os.path.join(path_result,'wGT')]

    all_TPs, all_FPs, all_FNs = 0, 0, 0    

    for i, name in enumerate(test_names):
        # loading data
        print(f'\n# No. {name}')
        print(f'\tLoading data...')
        img, dcm_file = get_image.ImgFromDcm(name, saveimg=True)

        test_x, patch_nameList, gt_boxes = get_image.get_patches(img.copy(), name, reSize)
        test_x= test_x/test_x.max()

        # predicting results
        print(f'\tStart inference...')
        results = model.predict(test_x, batch_size=64, verbose= 1)
        abprob = results.max()
        # converting result to box
        results_binary = results.copy()
        results_binary[results>=prob_th] = 1.0
        results_binary[results<prob_th] = 0.0

        # Making result area
        patch_mask = np.zeros(img.shape[:2], dtype = np.uint8)
        for r in range(len(results_binary)):
            if results_binary[r]== 1.0:
                x_min = int(patch_nameList[r].split('_')[-2])
                y_min = int(patch_nameList[r].split('_')[-1])
                patch_mask[y_min:y_min+reSize, x_min:x_min+reSize] = 255
                
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (d_radius,d_radius))
        dilation = cv2.dilate(patch_mask, kernel, iterations=4)

        # counting result boxes

        _, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 2)
        predict_boxes = np.ndarray((len(contours),4), dtype= np.uint16)
        for cnt in range(len(contours)):
            squeeze_loc = np.squeeze(contours[cnt], axis=1)
            min_loc = np.min(squeeze_loc, axis=0)
            max_loc = np.max(squeeze_loc, axis=0)
            predict_boxes[cnt] = min_loc[0], min_loc[1], max_loc[0], max_loc[1]       

        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        img = check_imges(img, predict_boxes, gt_boxes, abprob= abprob, wGT=args.wGT)
        for sp, rimg  in enumerate(img):
            cv2.imwrite(os.path.join(path_saveImg[sp], f'{name}.png'), cv2.cvtColor(rimg,cv2.COLOR_BGR2RGB))

        _dcm_save(img[0], dcm_file, os.path.join(path_result,'dcm'), name)

        TPs, FPs, FNs = count_indicators(gt_boxes, predict_boxes, th_iou = 0.1)
        all_TPs +=TPs
        all_FPs +=FPs
        all_FNs +=FNs

    sens, prec, fppi = get_performance(all_TPs, all_FPs, all_FNs, len(test_names))

    testResults_Det = pd.DataFrame({'SENs':f"{sens:0.2f}", 
                                    'PRECs': f"{prec:0.2f}", 
                                    'FPPI': f"{fppi:0.2f}",
                                    'N_Images' : len(test_names)
                                    }, index=[f'Total'])

    print('\n'+'#'*35)
    print(f'\t Total Result')  
    print('='*35)
    print(testResults_Det)
    print('#'*35+'\n')

    testResults_Det.to_excel(os.path.join(path_result,'text', f'Test_DetResults.xlsx'),index=None)

if __name__ == '__main__':
    main()