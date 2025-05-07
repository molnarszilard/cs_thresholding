"""
Created on Mon Aug 19 14:29:33 2019
@author: chen chen
chen.chen.adl@gmail.com
"""

from kde_gpu import conditional_probability
from kde_gpu import nadaraya_watson
from scipy import stats
import pandas as pd
import cupy as cp
import numpy as np
import time
import argparse, os
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dir', default="/mnt/ssd1/datasets/vineyards/vineye_leaves/leaves_simple/",
                    help='the directory to the source files')
parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab, luv, hls, hsv, ycrcb')
args = parser.parse_args()

dlist=os.listdir(args.dir)
dlist.sort()
print("Reading train images...")
image_train_filtered=[]
nr_images = len(dlist)
c_images = 0
image_train_filtered=[]
white_lower_range = np.array([245, 245, 245])
white_upper_range = np.array([256, 256, 256])
black_lower_range = np.array([0, 0, 0])
black_upper_range = np.array([15, 15, 15])
min_log_likes=[]
for filename in dlist:
    if filename.endswith(".jpg") or filename.endswith(".png"):
        c_images+=1
        path = args.dir+filename
        image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        maskgt = cv2.imread(path.replace('images', 'masks'))
        maskwhite = cv2.inRange(image, white_lower_range, white_upper_range)        
        maskblack = cv2.inRange(image, black_lower_range, black_upper_range)
        mask = maskblack + maskwhite            
        if args.cs=="rgb":
            img = image
        elif args.cs=="lab":        
            img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif args.cs=="luv":
            img = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif args.cs=="hls":
            img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif args.cs=="hsv":
            img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif args.cs=="ycrcb":
            img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            print("Unknown color space.")
            exit()
        # maskblack = cv2.inRange(maskgt, black_lower_range, black_upper_range)
        # maskwhite = cv2.inRange(maskgt, white_lower_range, white_upper_range)
        # maskgt[maskblack == 0] = 255
        # maskgt[maskwhite == 0] = 0
        maskgt = np.where(maskgt<127,0,1)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]==0 and maskgt[i,j,0]>0:
                    image_train_filtered.append(img[i,j,:])
    else:
        continue
image_train_filtered = np.asarray(image_train_filtered)
print(image_train_filtered.shape)

# rv = stats.expon(0,1)

x = image_train_filtered

# density_real = rv.pdf(x)

# t1=time.time()
# kde_scipy=stats.gaussian_kde(x.T,bw_method='silverman')
# kde_scipy=kde_scipy(x.T)
# print(time.time()-t1)

t1=time.time()
kde_cupy=nadaraya_watson.pdf(cp.asarray(x[:,0].T),bw_method='silverman')
print(time.time()-t1)

# df = pd.DataFrame({'x1':x,'kde_scipy':kde_scipy,
#                    'kde_cupy':cp.asnumpy(kde_cupy).squeeze(),'real density':density_real})

# df['scipy_mean_absolute_error']=np.abs(df['kde_scipy']-df['real density'])
# df['cupy_mean_absolute_error']=np.abs(df['kde_cupy']-df['real density'])
# print(df.mean())


# rv = stats.truncnorm(-3,2,30,10)
# nsample=10000
# x = cp.asarray(rv.rvs(nsample))
# ycondx = cp.asarray(cp.random.rand(nsample))
# y = 10*(ycondx-0.5)+x

# cdf_conditional_real = ycondx
# df = pd.DataFrame({'y':cp.asnumpy(y),'x':cp.asnumpy(x),'real density':cp.asnumpy(cdf_conditional_real)})

# df['nadaraya watson']= conditional_probability.cdf(y,x)
# df['nw_error']=np.abs(df['nadaraya watson']-df['real density'])
# df.mean()