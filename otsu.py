import argparse
import cv2
import numpy as np
import os
import timeit

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='canopy segmentation on individual images')
    parser.add_argument('--input', dest='input', default='./dataset/input_images/aghi/', type=str, help='path to a single input image for evaluation')
    parser.add_argument('--pred_folder', dest='pred_folder', default='./dataset/predicted_images/', type=str, help='where to save the predicted images.')
    parser.add_argument('--save_type', dest='save_type', default="mask", type=str, help='do you want to save the masked image, the mask, or both: splash, mask, both')
    parser.add_argument('--dim', dest='dim', default=False, type=bool, help='dim the pixels that are not segmented, or leave them black?')
    parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab')
    args = parser.parse_args()
    return args

def process_image(directory, filename,counter):
    img = cv2.imread(directory+filename)
    if args.save_type in ['splash','both']:
        imgmasked = img.copy()
        # imgmasked = cv2.resize(imgmasked,(640,480))
        # imgmasked = np.moveaxis(imgmasked,-1,0)
    if args.cs=="rgb":
        combination = 6
        invch0=False
        invch1=False
        invch2=False
    elif args.cs=="lab":        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        invch0=False
        invch1=False
        invch2=False
        combination = 6
    elif args.cs=="luv":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        invch0=False
        invch1=False
        invch2=False
        combination = 6
    elif args.cs=="hls":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        invch0=True
        invch1=False
        invch2=False
        combination = 6
    elif args.cs=="hsv":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        invch0=True
        invch1=False
        invch2=False
        combination = 6
    elif args.cs=="ycrcb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        invch0=False
        invch1=False
        invch2=False
        combination = 6
    else:
        print("Unknown color space.")
        exit()
    # img = cv2.resize(img,(640,480))
    # img = np.moveaxis(img,-1,0)
    # if args.cs=="rgb":
    # img/=255
    # print(img.shape)
    img_ch0 = img[:,:,0]
    img_ch1 = img[:,:,1]
    img_ch2 = img[:,:,2]
    if counter==0:
        start = timeit.default_timer()
        retval,maskpred0 = cv2.threshold(img_ch0,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        retval,maskpred1 = cv2.threshold(img_ch1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        retval,maskpred2 = cv2.threshold(img_ch2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        stop = timeit.default_timer()
        setuptime = stop-start
    else:
        setuptime = 0
    start = timeit.default_timer()
    retval,maskpred0 = cv2.threshold(img_ch0,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    retval,maskpred1 = cv2.threshold(img_ch1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    retval,maskpred2 = cv2.threshold(img_ch2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if invch0:
        maskpred0[maskpred0>0]=2
        maskpred0[maskpred0==0]=1
        maskpred0[maskpred0==2]=0
    if invch1:
        maskpred1[maskpred1>0]=2
        maskpred1[maskpred1==0]=1
        maskpred1[maskpred1==2]=0
    if invch2:
        maskpred2[maskpred2>0]=2
        maskpred2[maskpred2==0]=1
        maskpred2[maskpred2==2]=0
    if combination==0:
        maskpred=maskpred0
    elif combination==1:
        maskpred=maskpred1
    elif combination==2:
        maskpred=maskpred2
    elif combination==3:
        maskpred=np.round((maskpred0+maskpred1)//2,0)
    elif combination==4:
        maskpred=np.round((maskpred0+maskpred2)//2,0)
    elif combination==5:
        maskpred=np.round((maskpred1+maskpred2)//2,0)
    elif combination==6:
        maskpred=np.round((maskpred0+maskpred1+maskpred2)//3,0)
    # maskpred = np.where(maskpred<0.5,0,255)
    # print(maskpred.mean())
    # maskpred=maskpred.repeat(1,3,1,1)
    # print(maskpred.shape)
    stop = timeit.default_timer()
    # if args.cs=="lab":        
    #     maskpred = cv2.cvtColor(maskpred, cv2.COLOR_LAB2BGR)
    # elif args.cs=="luv":
    #     maskpred = cv2.cvtColor(maskpred, cv2.COLOR_LUV2BGR)
    # elif args.cs=="hls":
    #     maskpred = cv2.cvtColor(maskpred, cv2.COLOR_HLS2BGR)
    # elif args.cs=="hsv":
    #     maskpred = cv2.cvtColor(maskpred, cv2.COLOR_HSV2BGR)
    # elif args.cs=="ycrcb":
    #     maskpred = cv2.cvtColor(maskpred, cv2.COLOR_YCrCb2BGR)
    # if args.save_type in ['splash','both']:
    #     threshold = maskpred.mean()
    #     # masknorm = maskpred.copy()
    #     # masknorm[maskpred>=threshold]=0
    #     # masknorm[maskpred<threshold]=0
    #     # masknorm3=masknorm.repeat(1,1,1,1)
    #     if args.dim:
    #         imgmasked[maskpred<threshold]/=3
    #     else:
    #         imgmasked[maskpred<threshold]=0
    #     save_path=args.pred_folder+filename[:-4]
    #     outimage = imgmasked[0].cpu().detach().numpy()
    #     # outimage = np.moveaxis(outimage,0,-1)
    #     cv2.imwrite(save_path+'_pred_masked.jpg', outimage)
    if args.save_type in ['mask','both']:                
        save_path=args.pred_folder+filename[:-4]
        cv2.imwrite(save_path +'.jpg', maskpred)
    return start,stop,setuptime

if __name__ == '__main__':

    args = parse_args()
    isExist = os.path.exists(args.pred_folder)
    if not isExist:
        os.makedirs(args.pred_folder)
        print("The new directory for saving images while training is created!")
    
    print('evaluating...')

    if args.input.endswith('.png') or args.input.endswith('.jpg'):
        directory, filename = os.path.split(args.input)
        if not os.path.exists(args.input):
            print("The file: "+args.input+" does not exists.")
            exit()
        start,stop,_ = process_image(directory, filename,1)
        print('Predicting the image took %f seconds (with setup time)'% (stop-start))
    else:
        if os.path.isfile(args.input):
            print("The specified file: "+args.input+" is not an jpg or png image, nor a folder containing jpg or png images. If you want to evaluate videos, use eval_video.py or demo_video.py.")
            exit()
        if not os.path.exists(args.input):
            print("The folder: "+args.input+" does not exists.")
            exit()
        dlist=os.listdir(args.input)
        dlist.sort()
        time_sum = 0
        counter = 0
        for filename in dlist:
            if filename.endswith(".png") or filename.endswith(".jpg"):
                print("Predicting for:"+filename)
                start,stop,setuptime = process_image(args.input,filename,counter)
                if counter==0:
                    time_sum=stop-start
                    wsetuptime=setuptime
                else:
                    time_sum+=stop-start
                    wsetuptime+=stop-start
                counter=counter+1
            else:
                continue
        if counter==0:
            print("The specified folder: "+args.input+" does not contain images.")
        else:
            print('Predicting %d images took %f seconds, with the average of %f ( with setup time: %f, average: %f)' % (counter,time_sum,time_sum/counter,wsetuptime,wsetuptime/counter))  
    