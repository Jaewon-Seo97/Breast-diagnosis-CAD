import numpy as np
import cv2

import tensorflow as tf
import matplotlib.cm as cm


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

def binary_mask(_array, th):
    array = _array.copy()
    array[array>=th] = 255
    array[array<th] = 0
    
    return array.astype(np.uint8)

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

def mask2rectangle(mask, d_radius, th_prob=0.5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (d_radius,d_radius))
    dilation = cv2.dilate(mask, kernel, iterations=5)
    binary_dilation = binary_mask(dilation, th_prob)

    try:
        contours, _ = cv2.findContours(binary_dilation, cv2.RETR_EXTERNAL, 2)
    except ValueError:
        _, contours, _ = cv2.findContours(binary_dilation, cv2.RETR_EXTERNAL, 2)
    
    predict_boxes = np.ndarray((len(contours),4), dtype= np.uint16)
    
    for cnt in range(len(contours)):
        squeeze_loc = np.squeeze(contours[cnt], axis=1)
        min_loc = np.min(squeeze_loc, axis=0)
        max_loc = np.max(squeeze_loc, axis=0)
        predict_boxes[cnt] = min_loc[0], min_loc[1], max_loc[0], max_loc[1]

    return predict_boxes

def get_performance(all_TPs, all_FPs, all_FNs, len_img):
    sensitivity= (all_TPs+0.0001)/(all_TPs+all_FNs+0.0001)
    prec = (all_TPs+0.0001)/(all_TPs+all_FPs+0.0001)
    fppi = all_FPs/len_img    
    
    return sensitivity, prec, fppi

def trun_n_d(n,d):
    return (  n if not n.find('.')+1 else n[:n.find('.')+d+1]  )


def mod_outScale(img_shape, box):
    # box = [x_min, y_min, x_max, y_max]
    for b, loc in enumerate(box):
        if b < 2:
            mod = 0 if loc < 0 else loc
        else:
            ck_loc = img_shape[1] if b == 2 else img_shape[0]
            mod = loc if loc < ck_loc else ck_loc
        box[b] = mod
    return box

def recon_originSize(image, one_fileGt, result_boxes, BRlocList, reSize=512, wGT = True):
    crop_xmin = BRlocList[1]
    crop_ymin = BRlocList[0] 

    breast_region_shape_x = int(BRlocList[3]- BRlocList[1])
    breast_region_shape_y = int(BRlocList[2]- BRlocList[0])  

    r_stand = max(int(breast_region_shape_x), int(breast_region_shape_y))    
    ratio = r_stand/reSize

    half_padding_x = reSize/2 - (int(breast_region_shape_x)/r_stand)*reSize/2   
    half_padding_y = reSize/2 - (int(breast_region_shape_y)/r_stand)*reSize/2   
    half_padding_xmax = reSize/2 + (int(breast_region_shape_x)/r_stand)*reSize/2  
    half_padding_ymax = reSize/2 + (int(breast_region_shape_y)/r_stand)*reSize/2          

    if len(result_boxes) == 0:
        pass
    else:      
        for f in range(len(result_boxes)):
            xmin = int((result_boxes[f][0] - half_padding_x)*ratio + crop_xmin)
            ymin = int((result_boxes[f][1] - half_padding_y)*ratio + crop_ymin)
            xmax = int((result_boxes[f][2] - half_padding_x)*ratio + crop_xmin)
            ymax = int((result_boxes[f][3] - half_padding_y)*ratio + crop_ymin)
            mod_loc = mod_outScale(image.shape, [xmin, ymin, xmax, ymax])

            cv2.rectangle(image, (mod_loc[0], mod_loc[1]), (mod_loc[2], mod_loc[3]), (255,0,0), 6)

    if wGT == False:
        pass

    else:
        for f in one_fileGt.index.to_list():
            xmin = int((one_fileGt['x_min'][f] - half_padding_x)*ratio + crop_xmin)
            ymin = int((one_fileGt['y_min'][f] - half_padding_y)*ratio + crop_ymin)
            xmax = int((one_fileGt['x_max'][f] - half_padding_x)*ratio + crop_xmin)
            ymax = int((one_fileGt['y_max'][f] - half_padding_y)*ratio + crop_ymin)

            mod_loc = mod_outScale(image.shape, [xmin, ymin, xmax, ymax])

            cv2.rectangle(image, (mod_loc[0], mod_loc[1]), (mod_loc[2], mod_loc[3]), (0,255,0), 8)

    return image, [int(i) for i in [half_padding_x, half_padding_y, half_padding_xmax, half_padding_ymax]]


def make_gradcam_heatmap(img_array, model, last_conv_layer_name='onefinal_conv', pred_index = 0):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[0][:,:,:,pred_index]    

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(pred_target, heatmap, c_th = 0.9):

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("autumn_r")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap.copy()]
    jet_heatmap[pred_target<c_th] = 0


    return jet_heatmap


def blend_heat(box_img, BRlocList, resized_brheat, kinds, test_predict):
    blend = box_img[BRlocList[0]:BRlocList[2],BRlocList[1]:BRlocList[3]]
    blend_ = cv2.addWeighted(blend, 0.7, (resized_brheat*255).astype(np.uint8), 0.3, 0)
    box_img[BRlocList[0]:BRlocList[2], BRlocList[1]:BRlocList[3]] = blend_

    #######
    fsclae= 3
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    kind_text = f'{kinds[0]} score: {trun_n_d(str(test_predict[1][0][0]*100),1)}% | {kinds[1]} score: {trun_n_d(str(test_predict[1][0][1]*100),1)}%'
    fsize, BaseLine=cv2.getTextSize(kind_text,font,fsclae,2)
    korg = ((box_img.shape[1]//2)-fsize[0]//2, (box_img.shape[0] - 50))
    box_img = cv2.putText(box_img, kind_text, korg, font, 
                   fontScale=fsclae, color=(255,255,255), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
               
    return box_img

