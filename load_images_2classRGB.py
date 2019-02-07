import glob
from skimage import io , measure
import cv2
from scipy import ndimage
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import convolve2d as conv2
import sklearn.preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import numpy as np



class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super(MyLabelBinarizer, self).transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)

def sobel_op(image):
    dx = ndimage.sobel(image, 0)  # horizontal derivative
    dy = ndimage.sobel(image, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag = mag / np.max(mag)# normalize (Q&D)
    return mag
# construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")
 
# construct the Sobel y-axis kernel
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int")


def filter_images2D(img2D, filter1,filter2, stride):
    M1,N1 = img2D.shape
    M2,N2 = filter1.shape
    O1,O2 = (M1-M2+stride)//stride, (N1-N2+stride)//stride
    out = np.empty((O1,O2))

    out = conv2(img2D,filter1,'valid')[::stride,::stride]
    out2 = conv2(img2D,filter2,'valid')[::stride,::stride]
    
    return np.sqrt(out**2 + out2**2)

images_gecko = []
labels_gecko = []

images_baby = []
labels_baby = []

images_spider = []
labels_spider = []

images_noise = []
labels_noise = []

def load_data(testset_size):
    for filename in glob.glob('finalsample/0gecko/train-1353/*.bmp'): #assuming bmp
        image = io.imread(filename, as_grey=False)
        image = np.divide(image, 255.0)
        
        r_ = image[:,:,0]
        r_ = filter_images2D(r_, sobelY, sobelX, 3)
        r_ = np.clip(r_, 0, 1)
        g_ = image[:,:,1]
        g_ = filter_images2D(g_, sobelY, sobelX, 3)
        g_ = np.clip(g_, 0, 1)
        b_ = image[:,:,2]
        b_ = filter_images2D(b_, sobelY, sobelX, 3)
        b_ = np.clip(b_, 0, 1)

        if filename == 'finalsample/0gecko/train-1353/0_130.bmp' :
            printfile = np.stack((r_, g_, b_), axis=-1)
            plt.imshow(printfile)
            plt.savefig('0gecko_output.png')
            # plt.show()

        end = np.stack((r_, g_, b_), axis=0)
        end[(end > 1.0) | (end < 0.5)] = 0
        image = end.flatten()
        image = list(image)
        image = np.clip(image, 0, 1)
        image = list(image)
        image.append(1)
        images_gecko.append(image)
        labels_gecko.append(1)

    for filename in glob.glob('finalsample/1baby/train-1095/*.bmp'): #assuming bmp
        image = io.imread(filename, as_grey=False)
        image = np.divide(image, 255.0)

        r_ = image[:,:,0]
        r_ = filter_images2D(r_, sobelY, sobelX, 3)
        r_ = np.clip(r_, 0, 1)
        g_ = image[:,:,1]
        g_ = filter_images2D(g_, sobelY, sobelX, 3)
        g_ = np.clip(g_, 0, 1)
        b_ = image[:,:,2]
        b_ = filter_images2D(b_, sobelY, sobelX, 3)
        b_ = np.clip(b_, 0, 1)

        
        if filename == 'finalsample/1baby/train-1095/1_1024.bmp' :
            printfile = np.stack((r_, g_, b_), axis=-1)
            plt.imshow(printfile)
            plt.savefig('1baby_output.png')
            # plt.show()
        end = np.stack((r_, g_, b_), axis=0)
        end[(end > 1.0) | (end < 0.5)] = 0
        image = end.flatten()
        image = list(image)
        image = np.clip(image, 0, 1)
        image = list(image)
        image.append(1)
        images_baby.append(image)
        labels_baby.append(1)

    for filename in glob.glob('finalsample/2spider/train-1196/*.bmp'): #assuming bmp
        image = io.imread(filename, as_grey=False)
        image = np.divide(image, 255.0)

        r_ = image[:,:,0]
        r_ = filter_images2D(r_, sobelY, sobelX, 3)
        r_ = np.clip(r_, 0, 1)
        g_ = image[:,:,1]
        g_ = filter_images2D(g_, sobelY, sobelX, 3)
        g_ = np.clip(g_, 0, 1)
        b_ = image[:,:,2]
        b_ = filter_images2D(b_, sobelY, sobelX, 3)
        b_ = np.clip(b_, 0, 1)

        
        if filename == 'finalsample/2spider/train-1196/2_28.bmp' :
            printfile = np.stack((r_, g_, b_), axis=-1)
            plt.imshow(printfile)
            plt.savefig('2spider_output.png')
            # plt.show()
        end = np.stack((r_, g_, b_), axis=0)
        end[(end > 1.0) | (end < 0.5)] = 0
        image = end.flatten()
        image = list(image)
        image = np.clip(image, 0, 1)
        image = list(image)
        image.append(1) 
        images_spider.append(image)
        labels_spider.append(1)

    for filename in glob.glob('finalsample/3noise/train-2885/*.bmp'): #assuming bmp
        image = io.imread(filename, as_grey=False)
        image = np.divide(image, 255.0)

        r_ = image[:,:,0]
        r_ = filter_images2D(r_, sobelY, sobelX, 3)
        r_ = np.clip(r_, 0, 1)
        g_ = image[:,:,1]
        g_ = filter_images2D(g_, sobelY, sobelX, 3)
        g_ = np.clip(g_, 0, 1)
        b_ = image[:,:,2]
        b_ = filter_images2D(b_, sobelY, sobelX, 3)
        b_ = np.clip(b_, 0, 1)

        

        if filename == 'finalsample/3noise/train-2885/3_206.bmp' :
            printfile = np.stack((r_, g_, b_), axis=-1)
            plt.imshow(printfile)
            plt.savefig('3noise_output.png')
            # plt.show()
        end = np.stack((r_, g_, b_), axis=0)
        end[(end > 1.0) | (end < 0.5)] = 0
        image = end.flatten()
        image = list(image)
        image = np.clip(image, 0, 1)
        image = list(image)
        image.append(1) 
        images_noise.append(image)
        labels_noise.append(0)

    print(labels_gecko[:1], labels_baby[:1], labels_spider[:1], labels_noise[:1])
    print(len(labels_gecko), len(labels_baby), len(labels_spider), len(labels_noise))
    # print(len(labels_gecko[0]))
    X = images_gecko[:1000] + images_baby[:1000] + images_spider[:1000] + images_noise[:3000]
    y = labels_gecko[:1000] + labels_baby[:1000] + labels_spider[:1000] + labels_noise[:3000]
    print(len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testset_size, random_state = 50)
    print(np.mean(y_train))
    lb = MyLabelBinarizer()

    # label_binarizer.fit(range(2))

    # label1 = lb.fit_transform(y_test)
    # print(y_train)
    y_train_hot = lb.fit_transform(y_train)
    y_test_hot = lb.fit_transform(y_test)

    y_test_hot_tuple = []
    
    for label in y_test_hot:
        # print(label[0])
        y_test_hot_tuple.append(list(label))

    # print(y_test_hot_tuple)
    y_train_hot_tuple = []

    for label in y_train_hot:
        y_train_hot_tuple.append(list(label))
        
    return X_train, X_test, y_train_hot_tuple, y_test_hot_tuple