import cv2
import numpy as np
import torch 
def img_crop(img_arr):
    # breakout
    return img_arr[55:-15, 15:-15, :]


def rgb2gray(rgb):
    image_data = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    #image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def totensor(img_arr):
    return torch.FloatTensor(img_arr.transpose((2, 0, 1))).unsqueeze(dim=0)


def resize(rgb , channel=1) :
    image_data = cv2.resize(rgb, (84, 84))
    image_data = np.reshape(image_data, (84, 84, channel))
    return image_data


def data_transform(x):
    # breakout 
    x = img_crop(x)
    x = rgb2gray(x)
    x = resize(x,1)
    x = np.expand_dims(x.transpose((2, 0, 1)),axis=0)
    return x