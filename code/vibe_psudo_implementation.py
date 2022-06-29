import cv2 as cv 
import numpy as np
from glob import glob
from tqdm import tqdm
import time

class ViBE_algorithm:
    """
    nbSamples: number of samples per pixel
    reqMatches: #_min
    radius : R
    R_factor : scaling radius
    subsampleFactor: amount of random subsamping
    """
    def __init__(self, nbSamples = 20, reqMatches = 2, radius = 20,R_factor = 1, subsamplingFactor = 16, isRGB = False):
        self.nbSamples = nbSamples
        self.regMatches = reqMatches
        self.radius = radius
        self.R_factor = R_factor
        self.subsamplingFactor = subsamplingFactor
        self.isRGB = isRGB
        self.samples = None
        
    def distanceL1(self, a, b):
        if(self.isRGB == False):
            return abs(a - b)
        else:
            res = 0
            for i in range(3):
                res += abs(a[i] - b[i])
            return res
            
    # init for all rgb and gray images
    def initial_background(self, Img):
        I_pad = np.pad(Img, 1, 'edge')
        
        height = I_pad.shape[0]
        width = I_pad.shape[1]
        # RBG image
        if(Img.ndim > 2 ):
            I_pad = I_pad[:,:, 1 : -1]
            channel = I_pad.shape[2]
            samples = np.zeros((height, width, channel, self.nbSamples))
            
            self.isRGB = True
            self.R_factor = 4.5
        else:
            channel = 1
            samples = np.zeros((height, width, self.nbSamples))
        
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                for n in range(self.nbSamples):
                    x, y = 0, 0
                    while(x == 0 and y == 0):
                        x = np.random.randint(-1, 2)
                        y = np.random.randint(-1, 2)
                    ri = i + x
                    rj = j + y          
                    if(channel > 1):
                        for c in range(channel):
                            samples[i, j, c, n] = I_pad[ri, rj, c]
                    else:
                        samples[i,j,n] = I_pad[ri, rj]
                        
                        
        samples = samples[1 : height - 1, 1 : width - 1]
        return samples
    

    
    def vibe_detection(self, Img, samples):        
        height = Img.shape[0]
        width = Img.shape[1]
        segMap = np.zeros((height, width)).astype(np.uint8)
        
        for i in range(height):
            for j in range(width):
                count, index, dist = 0, 0, 0
                while(count < self.regMatches and index < self.nbSamples):
                    if(self.isRGB == True):
                        # try:
                        dist = self.distanceL1(Img[i, j], samples[i, j, :, index])
                        # except:
                            # print(Img[i, j].shape)
                            # print(samples[i,j,:,index].shape)
                    else:
                        dist = self.distanceL1(Img[i, j], samples[i, j, index])
                    
                    if(dist < self.R_factor * self.radius):
                        count += 1
                    index += 1
                # pixel classification according to reqMatches
                if(count >= self.regMatches): # the pixel belong to background
                    # stores the result in the segmentation map
                    segMap[i, j] = 0
                    # gets a random number between 0 and subsamplingFactor-1
                    r = np.random.randint(0, self.subsamplingFactor)
                    # update of the current pixel model
                    if( r == 0):
                        r = np.random.randint(0, self.subsamplingFactor)
                        if(self.isRGB == True):
                            samples[i, j, :, r] = Img[i, j]
                        else:
                            samples[i, j, r] = Img[i, j]
                    # update of a neighboring pixel model
                    r = np.random.randint(0 , self.subsamplingFactor )
                    if( r == 0): # random subsampling
                        # chooses a neighboring pixel randomly
                        x, y = 0, 0
                        while( x == 0 and y == 0):
                            x = np.random.randint(-1, 2)
                            y = np.random.randint(-1, 2)
                        r = np.random.randint(0, self.nbSamples)
                        ri = i + x
                        rj = j + y 
                        
                        try:
                            samples[ri, rj, r] = Img[ri, rj]
                        except:
                            pass
                else: #  the pixel belongs to the foreground
                    segMap[i, j] = 255
         
        return segMap, samples
    
    def apply(self, Img):
        if (self.samples is None):
            self.samples = self.initial_background(Img)
        
        segMap, self.samples = self.vibe_detection(Img, self.samples)
        # post-process with median filter 3x3 kernal
        segMap = cv.medianBlur(segMap, 3)
        
        return segMap
        # pass

        
        