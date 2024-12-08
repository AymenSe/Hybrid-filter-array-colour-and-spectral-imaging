import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


from typing import Tuple, Union, Literal
import numpy as np
from colour.utilities import as_float_array, tsplit, validate_method


class BayerCFA:
    """
    Class for Bayer Color Filter Array (CFA) operations with extended patterns.
    """

    def __init__(self, pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG", "RGXB", "BGXR", "GRBX", "GBRX"] = "RGGB"):
        """
        Initialize the BayerCFA class with a given CFA pattern.

        Parameters:
        - pattern: The CFA pattern as a string.
        """
        self.pattern = validate_method(
            pattern.upper(),
            ("RGGB", "BGGR", "GRBG", "GBRG", "RGXB", "BGXR", "GRBX", "GBRX"),
            '"{0}" CFA pattern is invalid, it must be one of {1}!',
        )

    def masks(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate masks for the CFA pattern.

        Parameters:
        - shape: Tuple representing the height and width of the image.

        Returns:
        - A tuple of masks (R, G, B) for the CFA.
        """
        
        channels = {channel: np.zeros(shape, dtype=bool) for channel in "RGYB"}
        
        for channel, (y, x) in zip(self.pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
            channel = channel.upper()
            # if channel != "X":  # "X" represents dead pixels, leave mask zeros
            channels[channel][y::2, x::2] = 1
        return tuple(channels.values())

    def apply(self, HSI: np.ndarray) -> np.ndarray:
        """
        Perform mosaicing to generate the CFA image from an RGB image.

        Parameters:
        - RGB: The input RGB image as a NumPy array.

        Returns:
        - The mosaiced CFA image as a NumPy array.
        """
        HSI = as_float_array(HSI)
        
        if HSI.ndim == 3:
            print("HSI is RGB")
            R, G, B = tsplit(HSI)
            R_m, G_m, B_m = self.masks(HSI.shape[:2])   
            # CFA combines the filtered contributions
            r = R * R_m
            g = G * G_m
            b = B * B_m
            return (r, g, b)
        elif HSI.ndim == 4:
            print("HSI is RGYB")
            R, G, Y, B = tsplit(HSI)
            R_m, G_m, Y_m, B_m = self.masks(HSI.shape[:2])   
            r = R * R_m
            g = G * G_m
            y = Y * Y_m
            b = B * B_m
            return (r, g, y, b)
            
    
    def display(self, mosaic: Tuple[np.ndarray, ...]) -> np.ndarray:
        # Displaying the Bayer pattern image
        if len(mosaic) == 3:
            mosaic = mosaic[0] + mosaic[1] + mosaic[2]
        else:
            mosaic = mosaic[0] + mosaic[1] + mosaic[2] + mosaic[3]
        # plt.imshow(mosaic)
        # plt.title(f"{self.pattern} Pattern ({self.pattern})")
        # plt.axis('off')
        # plt.show()

        return mosaic
    
