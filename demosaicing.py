import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

class Demosaicing:
    def __init__(self, pattern="RGGB"):
        """
        Initialize the Demosaicing class with a specified Bayer pattern.
        
        Parameters:
        pattern (str): The Bayer pattern. Default is "RGGB".
        """
        if pattern != "RGGB":
            raise ValueError("Currently, only 'RGGB' pattern is supported.")
        self.pattern = pattern

    def bilinear(self, mosaic):
        """
        Apply bilinear interpolation to demosaic the image.
        
        Parameters:
        im (numpy.ndarray): Input image array of shape (height, width, 3).
        
        Returns:
        tuple: Interpolated red, green, and blue channels as 2D arrays.
        """
        r, g, b = mosaic

        # Green interpolation kernel
        k_g = 1 / 4 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        convg = convolve2d(g, k_g, 'same')
        g = g + convg

        # Red interpolation kernels
        k_r_1 = 1 / 4 * np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        convr1 = convolve2d(r, k_r_1, 'same')
        convr2 = convolve2d(r + convr1, k_g, 'same')
        r = r + convr1 + convr2

        # Blue interpolation kernels
        k_b_1 = 1 / 4 * np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        convb1 = convolve2d(b, k_b_1, 'same')
        convb2 = convolve2d(b + convb1, k_g, 'same')
        b = b + convb1 + convb2

        return r, g, b

    def apply(self, im):
        """
        Apply demosaicing to the input image.
        
        Parameters:
        im (numpy.ndarray): Input image array of shape (height, width, 3).
        
        Returns:
        numpy.ndarray: Demosaiced image of shape (height, width, 3).
        """
        r, g, b = self.bilinear(im)
        demosaic_arr = np.array([r, g, b])
        demosaic_arr_transformed = np.transpose(demosaic_arr, (1, 2, 0))
        return demosaic_arr_transformed

    def display(self, demosaiced_image):
        """
        Apply demosaicing and display the resulting image.
        
        Parameters:
        im (numpy.ndarray): Input image array of shape (height, width, 3).
        
        Returns:
        None
        """

        plt.figure(figsize=(10, 10))
        plt.imshow(demosaiced_image.astype(np.uint8))
        plt.axis('off')
        plt.title('Demosaiced Image')
        plt.show()
