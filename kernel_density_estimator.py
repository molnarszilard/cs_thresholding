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
parser.add_argument('--test_one', default=False,type=bool,
                    help='test only one image, or every one of them?')
parser.add_argument('--only_train', default=False,type=bool,
                    help='do you want to run only the train?')
parser.add_argument('--heatmap', default=False,type=bool,
                    help='save the heatmap?')
parser.add_argument('--detect_image', default=False,type=bool,
                    help='save the thresholding image about the errors?')
parser.add_argument('--save_lab', default=False,type=bool,
                    help='save the lab image?')
parser.add_argument('--save_orig', default=False,type=bool,
                    help='save the original image?')
parser.add_argument('--train', default=False,type=bool,
                    help='save the original image?')
parser.add_argument('--model_path', default="/mnt/ssd1/szilard/projects/kernel_density/models/kde_healthy_test.pkl",
                    help='Path to the model')
parser.add_argument('--model_path2', default="/mnt/ssd1/szilard/projects/kernel_density/models/kde_disease_v2.pkl",
                    help='Path to the model')
parser.add_argument('--second_model', default=False,type=bool,
                    help='Do you want to evaluate a second model?')
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

dir_indist=args.dir
dir_outdist=args.dir+"disease/test/"

if args.train or args.only_train:
    dlist=os.listdir(dir_indist+"train/")
    dlist.sort()
    print("Reading train images...")
    image_train_filtered=[]
    for filename in dlist:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # print(filename)
            image = cv2.imread(dir_indist+"train/"+filename,cv2.IMREAD_UNCHANGED)
            # image = image[75:175,75:175,:]
            maskwhite = cv2.inRange(image, white_lower_range, white_upper_range)        
            maskblack = cv2.inRange(image, black_lower_range, black_upper_range)
            mask = maskblack + maskwhite
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # image_train_filtered=[]
            # coordinates_train=[]
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i,j]==0:
                        image_train_filtered.append(image[i,j,:])
                        # coordinates_train.append([i,j])
            # image_train_filtered = np.asarray(image_train_filtered)
            # coordinates_train = np.asarray(coordinates_train)
            # print(image_train_filtered.shape)
            # break
            # images_train.append(image)
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

    print("Testing Kernel Density for training data, please wait...")
    
    log_dens_train = parrallel_score_samples(model, image_train_filtered)
    
    # log_dens_train = model.score_samples(image_train_filtered)
    # print(image_train_filtered.max(axis=0))
    # print(log_dens_train[np.argmax(image_train_filtered, axis=0)[0]])
    # print(image_train_filtered.mean(axis=0))
    # print(image_train_filtered.min(axis=0))
    # print(log_dens_train[np.argmin(image_train_filtered, axis=0)[0]])
    print(log_dens_train.shape)
    print(log_dens_train)
    print(log_dens_train.max())
    print(image_train_filtered[np.argmax(log_dens_train),:])
    print(log_dens_train.mean())
    print(log_dens_train.min())
    print(image_train_filtered[np.argmin(log_dens_train),:])
    min_log_like = log_dens_train.min()
    plt.subplot(311)
    plt.plot(image_train_filtered[:,0], np.exp(log_dens_train), 'ro', linestyle="None")
    if args.second_model:
        print("Testing Kernel Density from Model 2 for training data, please wait...")
        model2 = joblib.load(args.model_path2)
        log_dens_train2 = parrallel_score_samples(model2, image_train_filtered)
        plt.plot(image_train_filtered, log_dens_train2, 'bo', linestyle="None")
    plt.subplot(312)
    plt.plot(image_train_filtered[:,1], np.exp(log_dens_train), 'ro', linestyle="None")
    plt.subplot(313)
    plt.plot(image_train_filtered[:,2], np.exp(log_dens_train), 'ro', linestyle="None")
    print("Plotting")
    plt.show()



    print("saving KD model to: " + kd_savename)
    joblib.dump(model, kd_savename)
else:
    min_log_like = -7.3747
if not args.only_train:
    model = joblib.load(kd_savename)
    if args.second_model:
        min_log_like2=-7.3688
        model2 = joblib.load(args.model_path2)

    folder = "heatmaps/"
    if not os.path.exists(args.dir+folder):
        os.makedirs(args.dir+folder)
    print("Testing images, not used in training.")
    dlist=os.listdir(dir_indist+"test/")
    dlist.sort()
    for filename in dlist:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(filename)
            image = cv2.imread(dir_indist+"test/"+filename,cv2.IMREAD_UNCHANGED)
            # image = image[75:175,75:175,:]        
            maskwhite = cv2.inRange(image, white_lower_range, white_upper_range)
            maskblack = cv2.inRange(image, black_lower_range, black_upper_range)
            mask = maskblack + maskwhite
            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            image_train_filtered=[]
            coordinates_train=[]
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i,j]==0:
                        image_train_filtered.append(image_lab[i,j,:])
                        coordinates_train.append([i,j])
            # image[mask != 0] = [0, 0, 255]
            image_train_filtered = np.asarray(image_train_filtered)
            if image_train_filtered.shape[0]<1:
                    continue
            coordinates_train = np.asarray(coordinates_train)
            log_dens_train = parrallel_score_samples(model, image_train_filtered)
            print(log_dens_train.max())
            print(log_dens_train.mean())
            print(log_dens_train.min())
            if args.second_model:
                log_dens_train2 = parrallel_score_samples(model2, image_train_filtered)
                print(log_dens_train2.max())
                print(log_dens_train2.mean())
                print(log_dens_train2.min())
                # detect_image2 = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
                detect_image2 = image.copy()
            detect_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
            detect_image_heat = np.zeros((image.shape[0],image.shape[1]))
            for i in range(log_dens_train.shape[0]):
                if log_dens_train[i]<min_log_like:
                    detect_image[coordinates_train[i,0],coordinates_train[i,1]]=(255,255,255)
                if args.second_model:
                    if log_dens_train[i]>min_log_like and log_dens_train2[i]<min_log_like2:
                        detect_image2[coordinates_train[i,0],coordinates_train[i,1]]=(0,255,255)
                    if log_dens_train[i]<min_log_like and log_dens_train2[i]>min_log_like2:
                        detect_image2[coordinates_train[i,0],coordinates_train[i,1]]=(0,0,255)
                    if log_dens_train[i]<min_log_like and log_dens_train2[i]<min_log_like2:
                        detect_image2[coordinates_train[i,0],coordinates_train[i,1]]=(255,0,0)
                detect_image_heat[coordinates_train[i,0],coordinates_train[i,1]]=log_dens_train[i]
            if args.heatmap:
                fig = plt.figure(figsize=(image.shape[1]/my_dpi, image.shape[0]/my_dpi), dpi=my_dpi)
                plt.imshow(detect_image_heat, cmap='hot', interpolation='nearest')
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                heatmap = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            if args.save_orig:
                image_orig = image.copy()
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
            savename = args.dir+folder+filename[:-4]+"_density_outliers.jpg"
            cv2.imwrite(savename, out_image)
            if args.heatmap:
                plt.close(fig)
        else:
            continue

    # print("Testing diseased images.")
    # dlist=os.listdir(dir_outdist)
    # dlist.sort()
    # for filename in dlist:
    #     if filename.endswith(".jpg") or filename.endswith(".png"):
    #         print(filename)
    #         image = cv2.imread(dir_outdist+filename,cv2.IMREAD_UNCHANGED)
    #         # image = image[75:175,75:175,:]        
    #         maskwhite = cv2.inRange(image, white_lower_range, white_upper_range)
    #         maskblack = cv2.inRange(image, black_lower_range, black_upper_range)
    #         mask = maskblack + maskwhite
    #         image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #         image_test_filtered=[]
    #         coordinates_test=[]
    #         for i in range(mask.shape[0]):
    #             for j in range(mask.shape[1]):
    #                 if mask[i,j]==0:
    #                     image_test_filtered.append(image_lab[i,j,:])
    #                     coordinates_test.append([i,j])
    #         # image[mask != 0] = [0, 0, 255]
    #         image_test_filtered = np.asarray(image_test_filtered)
    #         if image_test_filtered.shape[0]<1:
    #                 continue
    #         coordinates_test = np.asarray(coordinates_test)
    #         detect_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    #         detect_image_heat = np.zeros((image.shape[0],image.shape[1]))
    #         log_dens_test = parrallel_score_samples(model, image_test_filtered)
    #         # log_dens_test = model.score_samples(image_test_filtered)
    #         # print(log_dens_test.shape)
    #         # print(log_dens_test)
    #         print(log_dens_test.max())
    #         print(log_dens_test.mean())
    #         print(log_dens_test.min())
    #         if args.second_model:
    #             log_dens_test2 = parrallel_score_samples(model2, image_test_filtered)
    #             print(log_dens_test2.max())
    #             print(log_dens_test2.mean())
    #             print(log_dens_test2.min())
    #             # detect_image2 = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    #             detect_image2 = image.copy()
    #         for i in range(log_dens_test.shape[0]):
    #             if log_dens_test[i]<min_log_like:
    #                 detect_image[coordinates_test[i,0],coordinates_test[i,1]]=(255,255,255)
    #             if args.second_model:
    #                 if log_dens_test[i]>min_log_like and log_dens_test2[i]<min_log_like2:
    #                     detect_image2[coordinates_test[i,0],coordinates_test[i,1]]=(0,255,255)
    #                 if log_dens_test[i]<min_log_like and log_dens_test2[i]>min_log_like2:
    #                     detect_image2[coordinates_test[i,0],coordinates_test[i,1]]=(0,0,255)
    #                 if log_dens_test[i]<min_log_like and log_dens_test2[i]<min_log_like2:
    #                     detect_image2[coordinates_test[i,0],coordinates_test[i,1]]=(255,0,0)
    #             detect_image_heat[coordinates_test[i,0],coordinates_test[i,1]]=log_dens_test[i]
    #         if args.heatmap:
    #             fig = plt.figure(figsize=(image.shape[1]/my_dpi, image.shape[0]/my_dpi), dpi=my_dpi)
    #             plt.imshow(detect_image_heat, cmap='hot', interpolation='nearest')
    #             fig.canvas.draw()
    #             data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #             heatmap = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #         if args.save_orig:
    #             image_orig = image.copy()
    #         contours = cv2.findContours(detect_image[:,:,0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    #         cv2.drawContours(image, contours, -1, (0, 0, 255), 1) 
    #         out_image = image
    #         if args.save_orig:
    #             out_image=np.concatenate((out_image,image_orig), axis=1)
    #         if args.save_lab:
    #             out_image=np.concatenate((out_image,image_lab), axis=1)
    #         if args.detect_image:
    #             out_image=np.concatenate((out_image,detect_image), axis=1)
    #         if args.heatmap:
    #             out_image=np.concatenate((out_image,heatmap), axis=1)
    #         if args.second_model:
    #             out_image=np.concatenate((out_image,detect_image2), axis=1)    
    #         savename = args.dir+folder+filename[:-4]+"_density_outliers.jpg"
    #         cv2.imwrite(savename, out_image)
    #         if args.heatmap:
    #             plt.close(fig)
    #     else:
    #         continue




