from re import sub
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from glob import glob
import os
# import pybgs as bgs 
from vibe_fast import vibe_gray
from vibe_psudo_implementation import ViBE_algorithm


def demo_video(path, bg_algo=vibe_gray(), gray=True):
    cap = cv.VideoCapture(path)
    bg_algo = bg_algo
    frame_index = 0
    segmentation_time = 0
    update_time = 0
    t1 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if gray == True:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if frame_index % 100 == 0:
            print('Frame number: %d' % frame_index)
        
        mask = bg_algo.apply(gray_frame)
        cv.imshow('frame', frame)
        cv.imshow('mask', mask)
        frame_index += 1
        if cv.waitKey(1) & 0xFF == ord('q'):                                 # Break while loop if video ends
            break

    cap.release()
    cv.waitKey()
    cv.destroyAllWindows()
    
def demo_frames(path, bg_algo=vibe_gray(), gray=True):
    list_frames = sorted(glob(path + '/input/*.jpg'))
    for path in list_frames:
        frame = cv.imread(path)
        if gray == True:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = bg_algo.apply(gray_frame)
        cv.imshow('frame', frame)
        cv.imshow('mask', mask)
        if cv.waitKey(1) & 0xFF == ord('q'):                                 # Break while loop if video ends
            break
        
def evaluate_frames(path, bg_algo=vibe_gray(), gray=True):
    list_frames = sorted(glob(path + '/input/*.jpg'))
    gts = sorted(glob(path + '/groundtruth/*.png'))
    print('evaluate on', path)
    
    ROI_begin = 0
    ROI_end = len(list_frames)
    
    if( os.path.isfile(path + '/temporalROI.txt')):
        temporalROI = [int(x) for x in open(path + '/temporalROI.txt').read().split(' ')]    
        ROI_begin = temporalROI[0]
        ROI_end = temporalROI[1]
    
    Space_ROI = False
    if ( os.path.isfile(path + '/ROI.bmp')):
            Space_ROI = True
            SROI = cv.imread(path + '/ROI.bmp')
            SROI_shape = SROI.shape
            SROI = np.all(SROI, axis = 2).astype(np.uint8)
    else:
            Space_ROI = False
        
    
    print('has_Space_ROI:', Space_ROI)
    print('ROI_begin:', ROI_begin)
    print('ROI_end:', ROI_end)
    
    tp = 0; fp = 0; tn = 0; fn = 0

    
    for i in tqdm( range (len(list_frames) )):
        j = list_frames[i]
        # print(j)
        image = cv.imread(j)
        
        if(Space_ROI == True and SROI_shape != image.shape):
            Space_ROI = False
            print('Space ROI is not same size with the input image')
        
        # cv.imshow('frame', image)
        
        if(gray == True):
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        mask = bg_algo.apply(image)
        # img_bgmodel = bg_algo.getBackgroundModel()
        # process_time = time.time() - start
        
        k = gts[i] 
        # print(k)         
        gt = cv.imread(k)
        cv.imshow('gt', gt)
        gt = np.all(gt, axis = 2).astype(np.uint8)
        
        if (i >= ROI_begin and i <= ROI_end ):
           
            
            if(Space_ROI == False):
                tp += np.sum(np.logical_and(gt,mask))
                tn += np.sum(np.logical_not(np.logical_or(gt,mask)))
                fn += np.sum(np.logical_and(gt, np.logical_xor(gt,mask)))
                fp += np.sum(np.logical_and(mask, np.logical_xor(gt,mask)).astype(np.uint8) )
            else:
                
                tp += np.sum(np.logical_and(SROI,np.logical_and(gt,mask)))
                tn += np.sum(np.logical_and(SROI,np.logical_not(np.logical_or(gt,mask))))
                fn += np.sum(np.logical_and(SROI,np.logical_and(gt, np.logical_xor(gt,mask))))
                fp += np.sum(np.logical_and(SROI,np.logical_and(mask, np.logical_xor(gt,mask))))
        
        # cv.imshow('img_bgmodel', img_bgmodel)
        # cv.imshow('mask',mask)
        # cv.waitKey(1)
    cv.destroyAllWindows()
    rec = tp/(tp+fn)
    pr = tp/(tp+fp)
    f1 = (2*pr*rec)/(pr+rec)
    sp = tn/(tn+fp)
    fpr = fp/(fp+tn)
    fnr = fn/(tn+fp)
    pwc = 100*(fn+fp)/(tp+fn+fp+tn)
    
    print('recall:', rec)
    print('precision:', pr)
    print('f1:', f1)
    print('specificity:', sp)
    print('false positive rate:', fpr)
    print('false negative rate:', fnr)
    print('pwc:', pwc)
    return rec, pr, f1, sp, fpr, fnr, pwc


def evaluate_CDnet2014(root_path, bg_algo=vibe_gray(), gray=True):
    sub_vid = []
    rec = []
    pr = []
    f1 = []
    sp = []
    fpr = []
    pwc = []
    
    list_path = sorted(glob(root_path +'/dataset/*'))
    for path in list_path:
        sub_video_list = sorted(glob(path + '/*'))
        for sub_video in sub_video_list:
            # print(sub_video)
            vid_name = sub_video.split('/')[-1].split('.')[0]
            confusion_matrix =  evaluate_frames(sub_video, vibe_gray(), gray)
            sub_vid.append(vid_name)
            rec.append(confusion_matrix[0])
            pr.append(confusion_matrix[1])
            f1.append(confusion_matrix[2])
            sp.append(confusion_matrix[3])
            fpr.append(confusion_matrix[4])
            pwc.append(confusion_matrix[5])
    
    df = pd.DataFrame({'sub_vid': sub_vid, 'rec': rec, 'pr': pr, 'f1': f1, 'sp': sp, 'fpr': fpr, 'pwc': pwc})
    df.to_csv(root_path + '/evaluation.csv', index=False)
            
if __name__ == '__main__':
    # demo_video('')
    demo_frames(r'E:\VHT\CDnet2012\dataset2012\dataset\baseline\highway')
    # evaluate_frames(r'E:\VHT\CDnet2012\dataset2012\dataset\baseline\highway')
    # evaluate_CDnet2014('E:\VHT\CDnet2012\dataset2012')