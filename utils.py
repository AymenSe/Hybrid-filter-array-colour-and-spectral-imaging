import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def save_image(
    image: np.ndarray,
    filename: str,
    directory: str = "images",
    format: str = "png"
) -> str:
    """
    Saves an image to a specified file.

    Parameters:
    - image (np.ndarray): The image array (shape: height x width x channels or grayscale).
    - filename (str): The name of the file (without extension).
    - directory (str): The directory where the file will be saved. Default is "images".
    - format (str): The image format (e.g., "png", "jpeg"). Default is "png".
    
    Returns:
    - str: The full path of the saved image file.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Normalize floating-point image values if necessary
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() > 1.0:  # If values are in range [0, 255], normalize to [0, 1]
            image = image / 255.0

    # Construct the full path
    full_path = os.path.join(directory, f"{filename}.{format}")
    
    # Handle grayscale and RGB images
    if len(image.shape) == 2:  # Grayscale
        plt.imsave(full_path, image, cmap="gray", format=format)
    elif len(image.shape) == 3:  # RGB or RGBA
        plt.imsave(full_path, image, format=format)
    else:
        raise ValueError("Invalid image shape. Expected 2D (grayscale) or 3D (RGB).")
    
    print(f"Image saved at: {full_path}")
    return full_path


