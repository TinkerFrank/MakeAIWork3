import io
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
#pip install scikit-image
from skimage.measure import block_reduce

class ImageFilter(object):
    
    def __init__(self, kernel):
        self.imgKernel = kernel
    
    def convolve(self, imgTensor): #scipy not torch?
        imgTensorRGB = imgTensor.copy() 
        outputImgRGB = np.empty_like(imgTensorRGB)

        for dim in range(imgTensorRGB.shape[-1]):  # loop over rgb channels
            outputImgRGB[:, :, dim] = sp.signal.convolve2d (
                imgTensorRGB[:, :, dim], self.imgKernel, mode="same", boundary="symm"
            )

        return outputImgRGB        
    
    def downsample(self,imgTensor):
        imgTensorRGB = imgTensor.copy()
        downsizedimg = block_reduce(imgTensorRGB,block_size=(2,2,1), func=np.mean) #blocksize=scaling factor along each axis, return value for each block is the mean function
        return downsizedimg