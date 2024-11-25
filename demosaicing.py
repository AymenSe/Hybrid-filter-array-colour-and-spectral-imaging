import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from typing import Tuple

class Demosaicing:
    def __init__(self, config) -> None:
        """
        Initialize the Demosaicing class with a specified Bayer pattern.
        
        Parameters:
        pattern (str): The Bayer pattern. Default is "RGGB".
        """
        self.config = config
        pattern = self.config.pattern
        possible_patterns = ["RGGB", "RGXB"]
        if pattern not in possible_patterns:
            raise ValueError(f"Currently, {pattern} pattern is not supported.")

    def green_correction_bilinear(self, mosaic: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r, p_g, b = mosaic

        # Green interpolation kernel
        k_x = 1 / 4 * np.array([
            [1, 0, 1], 
            [0, 4, 0], 
            [1, 0, 1]])
        g = convolve2d(p_g, k_x, 'same')
        
        mosaic = (r, g, b)
        
        return self.bilinear(mosaic)
    
    def bilinear(self, mosaic: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply bilinear interpolation to demosaic the image.
        
        Parameters:
        mosaic (Tuple[np.ndarray, np.ndarray, np.ndarray]): 
            A tuple of three 2D numpy arrays representing the red, green, and blue channels of the mosaic.
        
        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            Interpolated red, green, and blue channels as 2D numpy arrays.
        """
        r, g, b = mosaic

        # print(g[:4, :4])
        
        # Green interpolation kernel
        k_g = 1 / 4 * np.array([
            [0, 1, 0], 
            [1, 0, 1], 
            [0, 1, 0]])
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

    def apply(self, im: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Apply demosaicing to the input image.
        
        Parameters:
        im (Tuple[np.ndarray, np.ndarray, np.ndarray]): 
            A tuple of three 2D numpy arrays representing the red, green, and blue channels of the mosaic.
        
        Returns:
        np.ndarray: 
            Demosaiced image as a 3D numpy array of shape (height, width, 3).
        """
        if self.config.demosaic_method == "bilinear":
            r, g, b = self.bilinear(im)
        elif self.config.demosaic_method == "green_correction_bilinear":
            r, g, b = self.green_correction_bilinear(im)
        else:
            raise ValueError(f"Currently, {self.config.demosaic_method} method is not supported.")
        
        demosaic_arr = np.array([r, g, b])
        demosaic_arr_transformed = np.transpose(demosaic_arr, (1, 2, 0))
        return demosaic_arr_transformed

    def display(self, demosaiced_image: np.ndarray) -> None:
        """
        Display the demosaiced image.
        
        Parameters:
        demosaiced_image (np.ndarray): 
            Demosaiced image as a 3D numpy array of shape (height, width, 3).
        
        Returns:
        None
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(demosaiced_image.astype(np.uint8))
        plt.axis('off')
        plt.title('Demosaiced Image')
        plt.show()
