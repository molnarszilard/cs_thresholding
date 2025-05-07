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
    parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab')
    parser.add_argument('--mode', dest='mode', default=1, type=int, help='default channels, inverted channels, or invert specific channels: 1,2,3')
    args = parser.parse_args()
    return args

def iou(pred, gt):
    intersection_tensor=pred*gt
    intersection = np.sum(intersection_tensor)
    union_tensor = pred+gt-intersection_tensor
    union = np.sum(union_tensor)
    loss = intersection/union
    return loss

def process_image(directory, filename):
    path = directory+filename
    img = cv2.imread(path)#.astype(np.uint8)
    # img = cv2.resize(img,(640,480))
    if args.cs=="rgb":
         path = path.replace('images', 'masks')
    elif args.cs=="lab":        
        path = path.replace('images_lab', 'masks')
    elif args.cs=="luv":
        path = path.replace('images_luv', 'masks')
    elif args.cs=="hls":
        path = path.replace('images_hls', 'masks')
    elif args.cs=="hsv":
        path = path.replace('images_hsv', 'masks')
    elif args.cs=="ycrcb":
        path = path.replace('images_ycrcb', 'masks')
    else:
        print("Unknown color space.")
        exit()
    maskgt = cv2.imread(path)[:,:,0]
    # maskgt = cv2.resize(maskgt,(640,480))
    # maskblack = cv2.inRange(maskgt, 0, 127)
    # maskwhite = cv2.inRange(maskgt, 127, 256)
    # maskgt[maskblack == 0] = 1
    # maskgt[maskwhite == 0] = 0
    maskgt = np.where(maskgt<127,0,1)
    if args.mode==1:
        invch0=False
        invch1=False
        invch2=False
    elif args.mode==2:
        invch0=True
        invch1=True
        invch2=True
    elif args.mode==3:
        if args.cs=="rgb":
            invch0=False
            invch1=False
            invch2=False
        elif args.cs=="lab":        
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            invch0=False
            invch1=False
            invch2=False
        elif args.cs=="luv":
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            invch0=False
            invch1=True
            invch2=False
        elif args.cs=="hls":
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            invch0=False
            invch1=False
            invch2=False
        elif args.cs=="hsv":
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            invch0=True
            invch1=False
            invch2=False
        elif args.cs=="ycrcb":
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            invch0=False
            invch1=True
            invch2=True
        else:
            print("Unknown color space.")
            exit()
    else:
        print("Unknown mode.")
        exit()
    # img = np.moveaxis(img,-1,0)
    # if args.cs=="rgb":
    # img/=255
    # print(img.dtype)
    img_ch0 = img[:,:,0]
    img_ch1 = img[:,:,1]
    img_ch2 = img[:,:,2]
    # print(img_ch0.shape)
    retval,maskpred0 = cv2.threshold(img_ch0,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    retval,maskpred1 = cv2.threshold(img_ch1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    retval,maskpred2 = cv2.threshold(img_ch2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    maskpred0[maskpred0>0]=1
    maskpred1[maskpred1>0]=1
    maskpred2[maskpred2>0]=1
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
    diff0 = (maskgt+1)-maskpred0
    diff1 = (maskgt+1)-maskpred1
    diff2 = (maskgt+1)-maskpred2
    diff3 = (maskgt+1)-np.round((maskpred0+maskpred1)//2,0)
    diff4 = (maskgt+1)-np.round((maskpred0+maskpred2)//2,0)
    diff5 = (maskgt+1)-np.round((maskpred1+maskpred2)//2,0)
    diff6 = (maskgt+1)-np.round((maskpred0+maskpred1+maskpred2)//3,0)
    pixels = maskgt.shape[0]*maskgt.shape[1]
    matches0 = np.where(diff0==1,1,0)
    matches1 = np.where(diff1==1,1,0)
    matches2 = np.where(diff2==1,1,0)
    matches3 = np.where(diff3==1,1,0)
    matches4 = np.where(diff4==1,1,0)
    matches5 = np.where(diff5==1,1,0)
    matches6 = np.where(diff6==1,1,0)
    accuracy0 = np.sum(matches0)/pixels
    accuracy1 = np.sum(matches1)/pixels
    accuracy2 = np.sum(matches2)/pixels
    accuracy3 = np.sum(matches3)/pixels
    accuracy4 = np.sum(matches4)/pixels
    accuracy5 = np.sum(matches5)/pixels
    accuracy6 = np.sum(matches6)/pixels
    # accuracy0 = iou(maskpred0,maskgt)
    # accuracy1 = iou(maskpred1,maskgt)
    # accuracy2 = iou(maskpred2,maskgt)
    # accuracy3 = iou((maskpred0+maskpred1)//2,maskgt)
    # accuracy4 = iou((maskpred0+maskpred2)//2,maskgt)
    # accuracy5 = iou((maskpred1+maskpred2)//2,maskgt)
    # accuracy6 = iou((maskpred0+maskpred1+maskpred2)//3,maskgt)
    
    print(accuracy0,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6)
    return accuracy0,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6

if __name__ == '__main__':

    args = parse_args()    
    print('estimating best combination...')
    if os.path.isfile(args.input):
        print("The specified file: "+args.input+" is not a folder.")
        exit()
    if not os.path.exists(args.input):
        print("The folder: "+args.input+" does not exists.")
        exit()
    dlist=os.listdir(args.input)
    dlist.sort()
    counter = 0
    ac0=0
    ac1=0
    ac2=0
    ac3=0
    ac4=0
    ac5=0
    ac6=0
    for filename in dlist:
        if filename.endswith(".png") or filename.endswith(".jpg"):
            print("Analyzing:"+filename)
            acc0,acc1,acc2,acc3,acc4,acc5,acc6 = process_image(args.input,filename)
            ac0+=acc0
            ac1+=acc1
            ac2+=acc2
            ac3+=acc3
            ac4+=acc4
            ac5+=acc5
            ac6+=acc6
            counter=counter+1
        else:
            continue
    if counter==0:
        print("The specified folder: "+args.input+" does not contain images.")
    else:
        print('The average accuracy for the different combinations:')
        print(ac0/counter)
        print(ac1/counter)
        print(ac2/counter)
        print(ac3/counter)
        print(ac4/counter)
        print(ac5/counter)
        print(ac6/counter)    