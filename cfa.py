import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class CFA:
    def __init__(self, pattern: str = "RGGB") -> None:
        """
        Initializes the CFA class with a specified Bayer pattern.

        Parameters:
        pattern (str): The Bayer pattern to use. Default is "RGGB".
        """
        possible_patterns = ["RGGB", "RGXB"]
        if pattern not in possible_patterns:
            raise ValueError(f"Currently, {pattern} pattern is not supported.")
        self.pattern = pattern

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Applies the CFA pattern to the input image to extract channels.

        Parameters:
        image (np.ndarray): 
            A 3-dimensional numpy array representing the input image with shape (height, width, 3).

        Returns:
        Tuple[np.ndarray, ...]: 
            A tuple of multiple 2-dimensional numpy arrays representing the channels.
        """
        if self.pattern == "RGGB":
            return rggb_filter_array(image)
        elif self.pattern == "RGXB":
            return rgxb_filter_array(image)

    def display(self, mosaic: Tuple[np.ndarray, ...]) -> np.ndarray:
        """
        Displays the result.

        Parameters:
        mosaic (Tuple[np.ndarray, ...]): 
            A tuple of multiple 2-dimensional numpy arrays representing the channels.

        Returns:
        np.ndarray: 
            A stacked image with shape (height, width, 3) for visualization.
        """
        # Create a copy of mosaic for processing
        mosaic_copy = [
            (channel / channel.max() * 255 if channel.max() > 0 else channel)
            for channel in mosaic
        ]

        # Stacking the channels for visualization
        stacked_image = np.stack(mosaic_copy, axis=2).astype(np.uint8)

        # Displaying the Bayer pattern image
        plt.imshow(stacked_image)
        plt.title(f"{self.pattern} Pattern ({self.pattern})")
        plt.axis('off')
        plt.show()

        return stacked_image


def rggb_filter_array(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies the RGGB Bayer pattern to the input image to extract red, green, and blue channels.

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

    return r, g, b


def rgxb_filter_array(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies the RGXB Bayer pattern to the input image to extract red, green, blue, and extra channels.

    Parameters:
    image (np.ndarray): 
        A 3-dimensional numpy array representing the input image with shape (height, width, 3).

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        A tuple of four 2-dimensional numpy arrays representing red, green, blue, and an extra channel.
    """


    # Initialize the channel arrays
    r = np.zeros(image.shape[:2])
    g = np.zeros(image.shape[:2])
    b = np.zeros(image.shape[:2])
    x = np.zeros(image.shape[:2])

    # Extract channels based on the RGXB Bayer pattern
    r[0::2, 0::2] = image[0::2, 0::2, 0]  # Red channel
    g[0::2, 1::2] = image[0::2, 1::2, 1]  # Green channel
    x[1::2, 0::2] = image[1::2, 1::2, 1]  # Extra channel (treated as green in this case)
    b[1::2, 1::2] = image[1::2, 1::2, 2]  # Blue channel

    return r, g, b


if __name__ == "__main__":
    # Load the image
    image = np.random.randint(254, 255, (6, 6, 3))

    # Initialize the CFA class
    cfa = CFA(pattern="RGGB")

    # Apply the Bayer pattern to the image
    mosaic = cfa.apply(image)
    
    # Display the result
    cfa.display(mosaic)
    
    cfax = CFA(pattern="RGXB")
    mosaicx = cfax.apply(image)
    cfa.display(mosaicx)
    
