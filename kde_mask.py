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
                    help='Do you want to train?')
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

dir_indist=args.dir

if args.train or args.only_train:
    dlist=os.listdir(dir_indist+"train/")
    dlist.sort()
    print("Reading train images...")
    image_train_filtered=[]
    nr_images = len(dlist)
    c_images = 0
    image_train_filtered=[]

    # model = joblib.load(kd_savename)
    min_log_likes=[]
    for filename in dlist:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # print(filename)
            
            c_images+=1
            path = dir_indist+"train/"+filename
            image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
            maskgt = cv2.imread(path.replace('images', 'masks'))
            # image = cv2.resize(image,(640,480))
            # maskgt = cv2.resize(maskgt,(640,480))
            # image = image[75:175,75:175,:]
            maskwhite = cv2.inRange(image, white_lower_range, white_upper_range)        
            maskblack = cv2.inRange(image, black_lower_range, black_upper_range)
            mask = maskblack + maskwhite            
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
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
            # image_train_filtered=[]
            # coordinates_train=[]
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i,j]==0 and maskgt[i,j,0]>0:
                        image_train_filtered.append(img[i,j,:])
                        # coordinates_train.append([i,j])
            # image_train_filtered = np.asarray(image_train_filtered)
            # coordinates_train = np.asarray(coordinates_train)
            # print(image_train_filtered.shape)
            # break
            # images_train.append(image)
            # if c_images>=nr_images/100000:                
            #     image_train_filtered = np.asarray(image_train_filtered)
            #     print(image_train_filtered.shape)
            #     log_dens_train = parrallel_score_samples(model, image_train_filtered)
            #     min_log_like= np.nanmin(log_dens_train[log_dens_train!=-np.inf])
            #     min_log_likes.append(min_log_like)
            #     print(min_log_like)
            #     c_images = 0
            #     image_train_filtered=[]
        else:
            continue
    # print("Minimum of minumus is:")
    # print(min_log_likes.min())
    image_train_filtered = np.asarray(image_train_filtered)
    print(image_train_filtered.shape)

    print("Calculating Kernel Density, please wait...")
    model = KernelDensity(kernel=kernels,bandwidth=bandwidth,atol=atols,rtol=rtols).fit(image_train_filtered)
    print("saving KD model to: " + kd_savename)
    joblib.dump(model, kd_savename)
    # model = joblib.load(kd_savename)
    # model = GridSearchCV(KernelDensity(),{'bandwidth': [bandwidth],'kernel':[kernels],'atol':[atols],'rtol':[rtols]},scoring=my_scores,verbose=10).fit(image_train_filtered)
    # model = model.best_estimator_
    # print("optimal kernel: " + "{}".format(model.kernel))
    # print("optimal bandwidth: " + "{:.4f}".format(model.bandwidth))
    # print("optimal atol: " + "{:.4f}".format(model.atol))
    # print("optimal rtol: " + "{:.4f}".format(model.rtol))

    print("Testing Kernel Density for training data, please wait...")
    # min_log_like=0.0
    log_dens_train = parrallel_score_samples(model, image_train_filtered)
            # len = image_train_filtered.shape[0]
            # for i in range(10):
            #     log_dens_train = parrallel_score_samples(model, image_train_filtered[i*int(len/10):(i+1)*int(len/10)-1,:])
    print(log_dens_train.shape)
            # print(log_dens_train)
            # print(log_dens_train.max())
            # print(image_train_filtered[np.argmax(log_dens_train),:])
            # print(log_dens_train.mean())
    min_log_like=np.nanmin(log_dens_train[log_dens_train!=-np.inf])
    print(min_log_like)
            # # print(image_train_filtered[np.argmin(log_dens_train),:])
            # min_log_like+= log_dens_train.min()
            # min_log_like/=10
            # plt.subplot(311)
            # plt.plot(image_train_filtered[:,0], np.exp(log_dens_train), 'ro', linestyle="None")
            # plt.subplot(312)
            # plt.plot(image_train_filtered[:,1], np.exp(log_dens_train), 'ro', linestyle="None")
            # plt.subplot(313)
            # plt.plot(image_train_filtered[:,2], np.exp(log_dens_train), 'ro', linestyle="None")
            # print("Plotting")
            # plt.show()



    
else:
    #min_log_like = -7.3747 #v1
    min_log_like = -10.590869388397683 #rgb - canopy
if not args.only_train:
    model = joblib.load(kd_savename)

    folder = "heatmaps/"
    if not os.path.exists(args.dir+folder):
        os.makedirs(args.dir+folder)
    print("Testing images, not used in training.")
    dlist=os.listdir(dir_indist+"test/")
    dlist.sort()
    for filename in dlist:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(filename)
            path = dir_indist+"test/"+filename
            image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
            maskgt = cv2.imread(path.replace('images', 'masks'))
            # image = cv2.resize(image,(640,480))
            # maskgt = cv2.resize(maskgt,(640,480))
            # image = image[75:175,75:175,:]        
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
            # image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            maskblack = cv2.inRange(maskgt, black_lower_range, black_upper_range)
            maskwhite = cv2.inRange(maskgt, white_lower_range, white_upper_range)
            maskgt[maskblack == 0] = 255
            maskgt[maskwhite == 0] = 0
            image_train_filtered=[]
            coordinates_train=[]
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i,j]==0:
                        image_train_filtered.append(img[i,j,:])
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
            detect_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
            detect_image_heat = np.zeros((image.shape[0],image.shape[1]))
            for i in range(log_dens_train.shape[0]):
                if log_dens_train[i]<min_log_like:
                    detect_image[coordinates_train[i,0],coordinates_train[i,1]]=(255,255,255)
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
                out_image=np.concatenate((out_image,img), axis=1)
            if args.detect_image:
                out_image=np.concatenate((out_image,detect_image), axis=1)
            if args.heatmap:
                out_image=np.concatenate((out_image,heatmap), axis=1)       
            savename = args.dir+folder+filename[:-4]+"_density_outliers.jpg"
            cv2.imwrite(savename, out_image)
            if args.heatmap:
                plt.close(fig)
        else:
            continue