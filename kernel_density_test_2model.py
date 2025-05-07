import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import argparse
import os
import cv2
import multiprocessing
import joblib
import timeit

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input', default="/mnt/ssd1/datasets/vineyards/vineye_leaves/leaves_simple/",
                    help='the directory to the source files')
parser.add_argument('--model_path', default="/mnt/ssd1/szilard/projects/kernel_density/models/kde_healthy_v1.pkl",
                    help='Path to the model')
parser.add_argument('--model_path2', default="/mnt/ssd1/szilard/projects/kernel_density/models/kde_healthy_v1.pkl",
                    help='Path to the model2')
parser.add_argument('--pred_folder', dest='pred_folder', default='./dataset/predicted_images/', type=str, help='where to save the predicted images.')
parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab, luv, hls, hsv, ycrcb')
args = parser.parse_args()

white_lower_range = np.array([245, 245, 245])
white_upper_range = np.array([256, 256, 256])
black_lower_range = np.array([0, 0, 0])
black_upper_range = np.array([15, 15, 15])
my_dpi = 300

def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))

model = KernelDensity()
model2 = KernelDensity()
kd_savename = args.model_path
print("Loading KD model from: " + kd_savename)
model = joblib.load(kd_savename)
kd_savename2 = args.model_path2
print("Loading KD model from: " + kd_savename2)
model2 = joblib.load(kd_savename2)
#min_log_like = -7.3747 #v1
# min_log_like = -10.590869388397683 #rgb
# min_log_like=-6.898794532770312 # kde_dis_leaf_d 
# min_log_like=-7.4759998731592265 # kde_h_leaf_labinfield
min_log_like=-7.375134886936371
min_log_like2=-6.898794532770312
def processing(directory,filename):
    print(filename)
    image = cv2.imread(directory+"/"+filename,cv2.IMREAD_UNCHANGED)      
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
    image_test_filtered=[]
    coordinates_test=[]
    maskwhite = cv2.inRange(image, white_lower_range, white_upper_range)
    maskblack = cv2.inRange(image, black_lower_range, black_upper_range)
    mask = maskblack + maskwhite
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j]==0:
                image_test_filtered.append(img[i,j,:])
                coordinates_test.append([i,j])
    image_test_filtered = np.asarray(image_test_filtered)
    coordinates_test = np.asarray(coordinates_test)
    detect_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    detect_image_heat = np.zeros((image.shape[0],image.shape[1]))
    print(image_test_filtered.shape)
    start = timeit.default_timer()
    log_dens_test = parrallel_score_samples(model, image_test_filtered)
    log_dens_test2 = parrallel_score_samples(model2, image_test_filtered)     
    stop = timeit.default_timer()
    for i in range(log_dens_test.shape[0]):
        if log_dens_test[i]<min_log_like and log_dens_test2[i]>=min_log_like2:
            detect_image[coordinates_test[i,0],coordinates_test[i,1]]=(255,255,255)
     
    savename = args.pred_folder+filename
    cv2.imwrite(savename, detect_image)
    return start, stop 
   
if args.input.endswith('.png') or args.input.endswith('.jpg'):
    directory, filename = os.path.split(args.input)
    if not os.path.exists(args.pred_folder):
        os.makedirs(args.pred_folder)
    start,stop = processing(directory, filename)
else:
    directory = args.input
    if not os.path.exists(args.pred_folder):
        os.makedirs(args.pred_folder)
    dlist=os.listdir(directory)
    dlist.sort()
    time_sum = 0
    counter = 0
    for filename in dlist:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            start,stop = processing(directory,filename)
            time_sum+=stop-start
            counter=counter+1
        else:
            continue
    if counter==0:
        print("The specified folder: "+args.input+" does not contain images.")
    else:
        print('Predicting %d images took %f seconds, with the average of %f' % (counter,time_sum,time_sum/counter))
