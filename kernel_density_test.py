import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import argparse
import os
import cv2
import multiprocessing
import joblib

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input', default="/mnt/ssd1/datasets/vineyards/vineye_leaves/leaves_simple/",
                    help='the directory to the source files')
parser.add_argument('--heatmap', default=False,type=bool,
                    help='save the heatmap?')
parser.add_argument('--detect_image', default=False,type=bool,
                    help='save the thresholding image about the errors?')
parser.add_argument('--save_lab', default=False,type=bool,
                    help='save the lab image?')
parser.add_argument('--save_orig', default=False,type=bool,
                    help='save the original image?')
parser.add_argument('--model_path', default="/mnt/ssd1/szilard/projects/kernel_density/models/kde_healthy_v1.pkl",
                    help='Path to the model')
parser.add_argument('--model_path2', default="/mnt/ssd1/szilard/projects/kernel_density/models/kde_disease_v2.pkl",
                    help='Path to the model')
parser.add_argument('--second_model', default=False,type=bool,
                    help='Do you want to evaluate a second model?')
parser.add_argument('--contour', default=False,type=bool,
                    help='draw the contour fill in the whole diseased area?')
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
kd_savename = args.model_path
print("Loading KD model from: " + kd_savename)
model = joblib.load(kd_savename)
min_log_like = -7.3747
if args.second_model:
    min_log_like2=-7.3688
    model2 = joblib.load(args.model_path2)

folder = "/heatmaps/"
threshold = 0.0006

def processing(directory,filename):
    print(filename)
    image = cv2.imread(directory+"/"+filename,cv2.IMREAD_UNCHANGED)      
    maskwhite = cv2.inRange(image, white_lower_range, white_upper_range)
    maskblack = cv2.inRange(image, black_lower_range, black_upper_range)
    mask = maskblack + maskwhite
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_test_filtered=[]
    coordinates_test=[]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j]==0:
                image_test_filtered.append(image_lab[i,j,:])
                coordinates_test.append([i,j])
    # image[mask != 0] = [0, 0, 255]
    image_test_filtered = np.asarray(image_test_filtered)
    coordinates_test = np.asarray(coordinates_test)
    detect_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    detect_image_heat = np.zeros((image.shape[0],image.shape[1]))
    print(image_test_filtered.shape)
    log_dens_test = parrallel_score_samples(model, image_test_filtered)     
    # print(log_dens_test.shape)
    # print(log_dens_test)
    # print(log_dens_test.max())
    # print(log_dens_test.mean())
    # print(log_dens_test.min())
    if args.save_orig:
        image_orig = image.copy()
    if args.second_model:
        log_dens_test2 = parrallel_score_samples(model2, image_test_filtered)
        # print(log_dens_test2.max())
        # print(log_dens_test2.mean())
        # print(log_dens_test2.min())
        # detect_image2 = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
        detect_image2 = image.copy()
    for i in range(log_dens_test.shape[0]):
        if log_dens_test[i]<min_log_like:
            if not args.contour:
                image[coordinates_test[i,0],coordinates_test[i,1]]=(0,0,255)
            detect_image[coordinates_test[i,0],coordinates_test[i,1]]=(255,255,255)
        if args.second_model:
            # if log_dens_test[i]>min_log_like and log_dens_test2[i]<min_log_like2:
            #     detect_image2[coordinates_test[i,0],coordinates_test[i,1]]=(0,255,255)
            # if log_dens_test[i]<min_log_like and log_dens_test2[i]>min_log_like2:
            #     detect_image2[coordinates_test[i,0],coordinates_test[i,1]]=(0,0,255)
            # if log_dens_test[i]<min_log_like and log_dens_test2[i]<min_log_like2:
            #     detect_image2[coordinates_test[i,0],coordinates_test[i,1]]=(255,0,0)
            if np.exp(log_dens_test[i])<threshold:
                if np.exp(log_dens_test2[i])>=np.exp(log_dens_test[i]) and np.exp(log_dens_test2[i])>=threshold:
                        detect_image2[coordinates_test[i,0],coordinates_test[i,1]]=(0,0,255)
                if np.exp(log_dens_test2[i])>=np.exp(log_dens_test[i]) and np.exp(log_dens_test2[i])<threshold:
                        detect_image2[coordinates_test[i,0],coordinates_test[i,1]]=(255,255,0)
                if np.exp(log_dens_test2[i])<np.exp(log_dens_test[i]):
                        detect_image2[coordinates_test[i,0],coordinates_test[i,1]]=(255,0,0)
            if np.exp(log_dens_test[i])>=threshold:
                if np.exp(log_dens_test2[i])>np.exp(log_dens_test[i]):
                        detect_image2[coordinates_test[i,0],coordinates_test[i,1]]=(0,255,255)
        detect_image_heat[coordinates_test[i,0],coordinates_test[i,1]]=log_dens_test[i]
    if args.heatmap:
        fig = plt.figure(figsize=(image.shape[1]/my_dpi, image.shape[0]/my_dpi), dpi=my_dpi)
        plt.imshow(detect_image_heat, cmap='hot', interpolation='nearest')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        heatmap = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if args.contour:
        contours = cv2.findContours(detect_image[:,:,0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image, contours, -1, (0, 0, 255), 1)
    out_image = image
    if args.save_orig:
        out_image=np.concatenate((out_image,image_orig), axis=1)
    if args.save_lab:
        out_image=np.concatenate((out_image,image_lab), axis=1)
    if args.detect_image:
        out_image=np.concatenate((out_image,detect_image), axis=1)
    if args.heatmap:
        out_image=np.concatenate((out_image,heatmap), axis=1)
    if args.second_model:
        out_image=np.concatenate((out_image,detect_image2), axis=1)
    # out_image=np.concatenate((image,image_lab,detect_image,heatmap), axis=1)
    
    savename = directory+folder+filename[:-4]+"_density_outliers.jpg"
    cv2.imwrite(savename, out_image)
    if args.heatmap:
        plt.close(fig)

if args.input.endswith('.png') or args.input.endswith('.jpg'):
    directory, filename = os.path.split(args.input)
    if not os.path.exists(directory+folder):
        os.makedirs(directory+folder)
    processing(directory, filename)
else:
    directory = args.input
    if not os.path.exists(directory+folder):
        os.makedirs(directory+folder)
    dlist=os.listdir(directory)
    dlist.sort()
    for filename in dlist:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            processing(directory,filename)
        else:
            continue
