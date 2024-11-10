#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:08:47 2024

@author: zsfatma
"""

#Import important libraries
import numpy as np
import spectral as sp #for read hyperspectral imagery
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
#import cv2



#Read hyperspectral imagery
hdr = sp.envi.open ("your_HSI_hdr_path_and_file.hdr") #choose .hdr file
wvl = hdr.bands.centers
rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
meta = hdr.metadata         
#Getting the HSI information
print ("HSI image metadata=", hdr)    #read metadata
print ("HSI range wavelength", wvl) #print all HSI wavelength
#Check your HSI file 
hdr.shape #(width, length, bands)



#Visualise HSI
img = hdr.load()
view = sp.imshow(img, ())


#Convert HSI into array type
img_arr = np.array(img)
type(img_arr)   #check


# Select three bands to create an RGB image (e.g., bands 30, 60, and 90)
rgb_image = np.stack((img_arr[:, :, 87], 
                      img_arr[:, :, 53], 
                      img_arr[:, :, 1]), axis=-1)


#Normalise the RGB image to the range [0, 1]
rgb_image = rgb_image / np.max(rgb_image)
rgb_image.shape     #check the HSI array is only contain 3 bands

# Normalise and convert to uint8 so the values are ranged from 0 to 255
def normalize_uint8(img, maxval, minval):
    """
    img: uint16 2d raw image
    out: uint8 2d normalized 0-255 image
    """
    return (np.rint((img - img.min()) * ((maxval - minval) / (img.max() - img.min())) + minval)).astype(dtype='uint8')

normalize_uint8(rgb_image,255,0)

#Bayer pattern CFA function (R G G B)
def bayer(im):
    """
    Decodes a 3-band hyperspectral image using the RGGB Bayer pattern to extract and separate the red, green, and blue channels.
    
    Parameters:
    im (numpy.ndarray): A 3-dimensional numpy array representing the input hyperspectral image with shape (height, width, 3).
    
    Returns:
    tuple: A tuple of three 2-dimensional numpy arrays representing the extracted red, green, and blue channels.
    """
    # Initialize the channel arrays
    r = np.zeros(im.shape[:2])
    g = np.zeros(im.shape[:2])
    b = np.zeros(im.shape[:2])
    
    # Extract the channels
    r[0::2, 0::2] = im[0::2, 0::2, 0]  # Red channel
    g[0::2, 1::2] = im[0::2, 1::2, 1]  # Green channel
    g[1::2, 0::2] = im[1::2, 0::2, 1]  # Green channel
    b[1::2, 1::2] = im[1::2, 1::2, 2]  # Blue channel
    
    return r, g, b

#Apply bayer function to HSI array
bayerimg = bayer(rgb_image)
bayerimg

# I tried to run but it shows black picture, maybe you have other options
def display_bayer_pattern(im):
    r, g, b = bayer(im)
    
    # Stacking the channels for visualization
    stacked_image = np.stack((r,g,b), axis=2).astype(np.uint8)
    
    # Displaying the Bayer pattern image
    plt.imshow(stacked_image)
    plt.title("Bayer Pattern")
    plt.axis('off')
    plt.show()



display_bayer_pattern(rgb_image)



#Demosaicing function
def bilinear(im):

    """
    Interpolate the channels of an image using bilinear interpolation based on a given Bayer pattern.

    Parameters:
    - im (numpy.ndarray): The image to be interpolated.

    Returns:
    - tuple: Interpolated red, green, and blue channels as 2D matrices.
    """
    # GREEN FIRST
    # Decode the Bayer pattern of the image to get the red, green, and blue channels.
    r, g, b = bayer(im)

    # Define a kernel for green interpolation
    k_g = 1/4 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    # Convolve the green channel with the kernel to get the interpolated values.
    convg =convolve2d(g, k_g, 'same')
    # Update the green channel with the interpolated values.
    g = g + convg

    # Define a kernel for initial red interpolation.
    k_r_1 = 1/4 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    # Convolve the red channel with the initial kernel.
    convr1 =convolve2d(r, k_r_1, 'same')
    # Perform a secondary convolution using the green kernel on the updated red channel.
    convr2 =convolve2d(r+convr1, k_g, 'same')
    # Update the red channel with both sets of interpolated values.
    r = r + convr1 + convr2

    # Define a kernel for initial blue interpolation (same as the red kernel).
    k_b_1 = 1/4 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    # Convolve the blue channel with the initial kernel.
    convb1 =convolve2d(b, k_b_1, 'same')
    # Perform a secondary convolution using the green kernel on the updated blue channel.
    convb2 =convolve2d(b+convb1, k_g, 'same')
    # Update the blue channel with both sets of interpolated values.
    b = b + convb1 + convb2
    # Return the interpolated red, green, and blue channels.
    return r, g, b



#Apply demosicing bilinear function
demosaic = bilinear(rgb_image)
demosaic_arr = np.array(demosaic)

# Convert from (3, 512, 512) to (512, 512, 3)
demosaic_arr_transformed = np.transpose(demosaic_arr, (1, 2, 0))
demosaic_arr_transformed


# Method 1: Using channels-first format directly
plt.figure(figsize=(10, 10))
plt.imshow(demosaic_arr_transformed)
plt.axis('off')  # Hide axes
plt.title('Demosaiced Image')
plt.show()








