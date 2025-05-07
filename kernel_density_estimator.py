import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import argparse
import os
import cv2
import multiprocessing
import joblib

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dir', default="/mnt/ssd1/datasets/vineyards/vineye_leaves/leaves_simple/",
                    help='the directory to the source files')
parser.add_argument('--model_path', default="/mnt/ssd1/szilard/projects/kernel_density/models/kde_healthy_test.pkl",
                    help='Path to the model')
parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab, luv, hls, hsv, ycrcb')
args = parser.parse_args()

mpl.rcParams['agg.path.chunksize'] = 10000000

kd_savename = args.model_path
kernels = ['cosine', 'epanechnikov', 'exponential', 'gaussian', 'linear', 'tophat']
bandwidth = np.arange(0.05, 2, .05)
atols = np.arange(0.0001, 0.001, .0005)
rtols = np.arange(0.001, 0.01, .005)
kernels = "linear"
bandwidth = 0.05
atols = 0.0001
rtols = 0.001
white_lower_range = np.array([245, 245, 245])
white_upper_range = np.array([256, 256, 256])
black_lower_range = np.array([0, 0, 0])
black_upper_range = np.array([15, 15, 15])
my_dpi=300


def my_scores(estimator, X):
    scores = estimator.score_samples(X)
    # Remove -inf
    scores = scores[scores != float('-inf')]
    scores = scores[scores != float('nan')]
    # Return the mean values
    return np.mean(scores)

def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))


dlist=os.listdir(args.dir)
dlist.sort()
print("Reading train images...")
image_train_filtered=[]
for filename in dlist:
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = cv2.imread(args.dir+filename,cv2.IMREAD_UNCHANGED)
        maskwhite = cv2.inRange(image, white_lower_range, white_upper_range)        
        maskblack = cv2.inRange(image, black_lower_range, black_upper_range)
        mask = maskblack + maskwhite
        if args.cs=="rgb":
            image = image
        elif args.cs=="lab":        
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif args.cs=="luv":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif args.cs=="hls":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif args.cs=="hsv":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif args.cs=="ycrcb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            print("Unknown color space.")
            exit()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i,j]==0:
                    image_train_filtered.append(image[i,j,:])
    else:
        continue
image_train_filtered = np.asarray(image_train_filtered)
print(image_train_filtered.shape)

print("Calculating Kernel Density, please wait...")
model = KernelDensity(kernel=kernels,bandwidth=bandwidth,atol=atols,rtol=rtols).fit(image_train_filtered)
# model = GridSearchCV(KernelDensity(),{'bandwidth': [bandwidth],'kernel':[kernels],'atol':[atols],'rtol':[rtols]},scoring=my_scores,verbose=10).fit(image_train_filtered)
# model = model.best_estimator_
# print("optimal kernel: " + "{}".format(model.kernel))
# print("optimal bandwidth: " + "{:.4f}".format(model.bandwidth))
# print("optimal atol: " + "{:.4f}".format(model.atol))
# print("optimal rtol: " + "{:.4f}".format(model.rtol))
print("saving KD model to: " + kd_savename)
joblib.dump(model, kd_savename)

print("Testing Kernel Density for training data, please wait...")

log_dens_train = parrallel_score_samples(model, image_train_filtered)
min_log_like=np.nanmin(log_dens_train[log_dens_train!=-np.inf])
print(min_log_like)
  