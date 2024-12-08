import numpy as np
import spectral as sp
from typing import Tuple, Dict, List

class HyperspectralImageProcessor:
    def __init__(self, hdr_file_path: str = None, img_file_path: str = None) -> None:
        """
        Initialize the processor with the .hdr file path.
        
        Parameters:
        hdr_file_path (str): Path to the .hdr file of the hyperspectral image.
        """
        self.hdr_file_path = hdr_file_path
        self.img_file_path = img_file_path
        self.hdr = sp.envi.open(hdr_file_path,  self.img_file_path if self.img_file_path else None)
        self.wvl: List[float] = self.hdr.bands.centers
        self.rows: int = self.hdr.nrows
        self.cols: int = self.hdr.ncols
        self.bands: int = self.hdr.nbands
        self.meta: Dict[str, str] = self.hdr.metadata
        self.img = self.hdr.load()

    def get_metadata(self) -> Dict[str, str]:
        """
        Returns the metadata of the HSI file.
        
        Returns:
        Dict[str, str]: Metadata dictionary.
        """
        return self.meta

    def get_wavelengths(self) -> List[float]:
        """
        Returns the wavelength information.
        
        Returns:
        List[float]: List of wavelengths.
        """
        return self.wvl

    def get_image_shape(self) -> Tuple[int, int, int]:
        """
        Returns the shape of the hyperspectral image.
        
        Returns:
        Tuple[int, int, int]: Shape of the image as (width, length, bands).
        """
        return self.hdr.shape

    def visualize_image(self) -> None:
        """
        Visualizes the hyperspectral image using Spectral's viewer.
        """
        sp.imshow(self.img, ())

    def to_array(self) -> np.ndarray:
        """
        Converts the hyperspectral image to a NumPy array.
        
        Returns:
        np.ndarray: NumPy array of the hyperspectral image.
        """
        return np.array(self.img)

    def create_rgb_image(self) -> np.ndarray:
        """
        Create an RGB image from selected bands.
        
        Parameters:
        band_indices (Tuple[int, int, int]): A tuple of three indices representing R, G, B bands.
        
        Returns:
        np.ndarray: Normalized RGB image as a NumPy array.
        """
        band_indices = list(self.meta["default bands"])
        band_indices = [int(band) for band in band_indices]

        if len(band_indices) != 3:
            raise ValueError("Three band indices are required to create an RGB image.")
        
        img_arr = self.to_array()
        rgb_image = np.stack(
            (img_arr[:, :, band_indices[0]],
             img_arr[:, :, band_indices[1]],
             img_arr[:, :, band_indices[2]]),
            axis=-1
        )
        return rgb_image / np.max(rgb_image)

    def create_rgyb_image(self, y_idx: int) -> np.ndarray:
        """
        Create an RGYB image from selected bands.
        
        Returns:
        np.ndarray: Normalized RGYB image as a NumPy array.
        """
        band_indices = list(self.meta["default bands"]) # RGB
        band_indices = [int(band) for band in band_indices] # [12, 8, 4]
        band_indices.insert(2, y_idx) # [12, 8, 6, 4]

        if len(band_indices) != 4:
            raise ValueError("Four band indices are required to create an RGYB image.")
        
        img_arr = self.to_array()
        
        rgyb_image = np.stack(
            (img_arr[:, :, band_indices[0]],
             img_arr[:, :, band_indices[1]],
             img_arr[:, :, band_indices[2]],
             img_arr[:, :, band_indices[3]]),
            axis=-1
        )
        return rgyb_image / np.max(rgyb_image)
    
    @staticmethod
    def normalize_uint8(img: np.ndarray, maxval: int = 255, minval: int = 0) -> np.ndarray:
        """
        Normalizes an image to the range [minval, maxval] and converts to uint8.
        
        Parameters:
        img (np.ndarray): Input image as a NumPy array.
        maxval (int): Maximum value for normalization (default 255).
        minval (int): Minimum value for normalization (default 0).
        
        Returns:
        np.ndarray: Normalized image as a uint8 NumPy array.
        """
        return (np.rint((img - img.min()) * ((maxval - minval) / (img.max() - img.min())) + minval)).astype(dtype='uint8')
