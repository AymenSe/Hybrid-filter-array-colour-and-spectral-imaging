import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class CFA:
    def __init__(self, pattern: str = "RGGB") -> None:
        """
        Initializes the CFA class with a specified Bayer pattern.

        Parameters:
        pattern (str): The Bayer pattern to use. Default is "RGGB".
        """
        if pattern != "RGGB":
            raise ValueError("Currently, only 'RGGB' pattern is supported.")
        self.pattern = pattern

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies the Bayer pattern to the input image to extract red, green, and blue channels.

        Parameters:
        image (np.ndarray): 
            A 3-dimensional numpy array representing the input image with shape (height, width, 3).

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            A tuple of three 2-dimensional numpy arrays representing the red, green, and blue channels.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must have 3 dimensions with shape (height, width, 3).")
        
        # Initialize the channel arrays
        r = np.zeros(image.shape[:2])
        g = np.zeros(image.shape[:2])
        b = np.zeros(image.shape[:2])
        
        # Extract channels based on the RGGB Bayer pattern
        r[0::2, 0::2] = image[0::2, 0::2, 0]  # Red channel
        g[0::2, 1::2] = image[0::2, 1::2, 1]  # Green channel
        g[1::2, 0::2] = image[1::2, 0::2, 1]  # Green channel
        b[1::2, 1::2] = image[1::2, 1::2, 2]  # Blue channel

        return (r, g, b)

    def display(self, mosaic: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Applies the Bayer pattern and displays the result.

        Parameters:
        mosaic (Tuple[np.ndarray, np.ndarray, np.ndarray]): 
            A tuple of three 2-dimensional numpy arrays representing the red, green, and blue channels.

        Returns:
        np.ndarray: 
            A 3-dimensional numpy array representing the stacked image with shape (height, width, 3).
        """
        r, g, b = mosaic
        # Normalize values to [0, 255] for visualization
        r = (r / r.max() * 255) if r.max() > 0 else r
        g = (g / g.max() * 255) if g.max() > 0 else g
        b = (b / b.max() * 255) if b.max() > 0 else b
        
        # Stacking the channels for visualization
        stacked_image = np.stack((r, g, b), axis=2).astype(np.uint8)
        
        # Displaying the Bayer pattern image
        plt.imshow(stacked_image)
        plt.title(f"Bayer Pattern ({self.pattern})")
        plt.axis('off')
        plt.show()

        return stacked_image
