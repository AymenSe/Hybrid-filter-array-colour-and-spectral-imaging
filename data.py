import numpy as np
import spectral as sp

class HyperspectralImageProcessor:
    def __init__(self, hdr_file_path):
        """
        Initialize the processor with the .hdr file path.
        :param hdr_file_path: Path to the .hdr file of the hyperspectral image.
        """
        self.hdr_file_path = hdr_file_path
        self.hdr = sp.envi.open(hdr_file_path)
        self.wvl = self.hdr.bands.centers
        self.rows, self.cols, self.bands = self.hdr.nrows, self.hdr.ncols, self.hdr.nbands
        self.meta = self.hdr.metadata
        self.img = self.hdr.load()

    def get_metadata(self):
        """
        Returns the metadata of the HSI file.
        :return: Metadata dictionary.
        """
        return self.meta

    def get_wavelengths(self):
        """
        Returns the wavelength information.
        :return: List of wavelengths.
        """
        return self.wvl

    def get_image_shape(self):
        """
        Returns the shape of the hyperspectral image.
        :return: Tuple (width, length, bands).
        """
        return self.hdr.shape

    def visualize_image(self):
        """
        Visualizes the hyperspectral image using Spectral's viewer.
        """
        sp.imshow(self.img, ())

    def to_array(self):
        """
        Converts the hyperspectral image to a NumPy array.
        :return: NumPy array of the hyperspectral image.
        """
        return np.array(self.img)

    def create_rgb_image(self, band_indices):
        """
        Create an RGB image from selected bands.
        :param band_indices: A tuple of three indices representing R, G, B bands.
        :return: Normalized RGB image as a NumPy array.
        """
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

    @staticmethod
    def normalize_uint8(img, maxval=255, minval=0):
        """
        Normalizes an image to the range [minval, maxval] and converts to uint8.
        :param img: Input image as a NumPy array.
        :param maxval: Maximum value for normalization (default 255).
        :param minval: Minimum value for normalization (default 0).
        :return: Normalized image as a uint8 NumPy array.
        """
        return (np.rint((img - img.min()) * ((maxval - minval) / (img.max() - img.min())) + minval)).astype(dtype='uint8')


