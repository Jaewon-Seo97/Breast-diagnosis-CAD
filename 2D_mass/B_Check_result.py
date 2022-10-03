import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from skimage import io
import cv2
import numpy as np
import time

import utils.data_utils as du
import utils.result_utils as ru
import utils.layer_utils as lu

from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

import argparse

   
def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default = 0)   
    return parser.parse_args(args)




def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    path_init = os.path.abspath('..')
    path_model = os.path.join(path_init, 'model', 'model_MGMass.h5')
    imageSize= 512
    d_radius = 12
    confidenc_th = 0.9
        
    savepath = os.path.join(path_init, f'result_last')
    for a in ['img', 'text', 'dcm', 'wGT']:
        os.makedirs(os.path.join(savepath, a), exist_ok=True)

    get_image = du._get_image(dcm_path = os.path.join(path_init, 'data'), r_erode=0, reSize = imageSize)
    get_custom_objects().update({
                                'swish':lu.Swish(lu.swish),
                                'Swish':lu.Swish(lu.swish),
                                'EfficientNetConvInitializer': lu.EfficientNetConvInitializer,
                                'EfficientNetDenseInitializer': lu.EfficientNetDenseInitializer,
                                    })
    
    print('\n'+'Loading model...')
    model = load_model(path_model, compile=False)
    print('\tModel load success')
    testnames = [name.split('.')[0] for name in os.listdir(os.path.join(path_init, 'data', 'dcm')) if name.endswith('.dcm')]
    path_gtLoc = os.path.join(path_init, 'data', 'DetEachGT_Location.xlsx')
    gtLocTextList = pd.read_excel(f'{path_gtLoc}', names = ['idx','machine','view','file_name', 'x_min', 'y_min', 'x_max', 'y_max'], index_col=None, engine = 'openpyxl')
    
    # loading data
    all_TPs, all_FPs, all_FNs = 0, 0, 0 
    kinds = {0: 'Benign', 
            1: 'Malignant', 
            2: 'Normal'}

    for i, name in enumerate(testnames):
        
        print(f'\n# No. {name}')
        print('\t'+f'Loading data...')
        one_fileGt = gtLocTextList[gtLocTextList['file_name']==f'{name[1:]}']
        gt_boxes = np.asarray(one_fileGt)[:,-4:]
        img, img_raw, dcmfile = get_image.ImgFromDcm(name, saveimg=True)
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        test_x, BRlocList= get_image.get_inputdata(name, img_raw, img.copy(), wGT=False)   #####

        print('\t'+f'Start prediction...')
        test_predict = model.predict(np.expand_dims(test_x/255.0, axis=0),batch_size=20, verbose=1)

        predict_boxes = ru.mask2rectangle(mask=ru.binary_mask(test_predict[0][0][:,:,0].copy(), 
                                                        th = confidenc_th), 
                                    d_radius = d_radius)

        heatmap = ru.make_gradcam_heatmap(img_array = np.expand_dims(test_x/255.0, axis=0), 
                                        model = model)   
        heat_arr = ru.save_and_display_gradcam(pred_target = test_predict[0][0][:,:,0],
                                                    heatmap= heatmap, c_th=confidenc_th)
        
        breast_region_shape_x = int(BRlocList[3]- BRlocList[1])
        breast_region_shape_y = int(BRlocList[2]- BRlocList[0]) 

        
        box_img, half_padding = ru.recon_originSize(color_img.copy(), one_fileGt, predict_boxes, BRlocList, reSize=512, wGT=False)
        
        resized_brheat = cv2.resize(heat_arr[half_padding[1]: half_padding[3],half_padding[0]:half_padding[2], :], 
                                (breast_region_shape_x, breast_region_shape_y))
        
        box_img = ru.blend_heat(box_img, BRlocList, resized_brheat, kinds, test_predict)
        io.imsave(os.path.join(savepath, 'img',f'{name}.png'), box_img)
        du._dcm_save(box_img, dcmfile, savepath, name)
        
        box_img, half_padding = ru.recon_originSize(color_img.copy(), one_fileGt, predict_boxes, BRlocList, reSize=512, wGT=True)
        box_img = ru.blend_heat(box_img, BRlocList, resized_brheat, kinds, test_predict)
        io.imsave(os.path.join(savepath, 'wGT',f'{name}.png'),box_img)

        TPs, FPs, FNs = ru.count_indicators(gt_boxes, predict_boxes, th_iou = 0.1)
        all_TPs +=TPs
        all_FPs +=FPs
        all_FNs +=FNs

    sens, prec, fppi = ru.get_performance(all_TPs, all_FPs, all_FNs, len(testnames))

    testResults_Det = pd.DataFrame({'SENs':f"{sens:0.2f}", 
                                    'PRECs': f"{prec:0.2f}", 
                                    'FPPI': f"{fppi:0.2f}",
                                    'N_Images' : len(testnames)
                                    }, index=[f'Total'])

    print('\n'+'#'*35)
    print(f'\t Total Result')  
    print('='*35)
    print(testResults_Det)
    print('#'*35+'\n')

    testResults_Det.to_excel(os.path.join(savepath,'text', f'Test_DetResults.xlsx'),index=None)


        
        
if __name__ == '__main__':
    main()