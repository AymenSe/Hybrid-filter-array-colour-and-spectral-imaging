from data import *
from cfa import *
from demosaicing import *
from reconstruction import *
from iqa import *
from utils import *


def main(config):
    processor = HyperspectralImageProcessor(config.hsi_path)
    cfa = CFA(config.pattern)
    demosaicer = Demosaicing(config.pattern)
    # Example usage:
    print("Loading hyperspectral image...")
    metadata = processor.get_metadata()
    wavelengths = processor.get_wavelengths()
    print ("HSI image metadata=", metadata)    #read metadata
    print ("HSI range wavelength", wavelengths) #print all HSI wavelength

    print("Creating RGB image...")
    rgb_image = processor.create_rgb_image(config.rgb_indices)
    normalized_rgb = processor.normalize_uint8(rgb_image)
    
    print("Applying CFA...")
    mosaic = cfa.apply(normalized_rgb)
    cfa.display(mosaic)
    
    print("Demosaicing RGB image...")
    demosaiced = demosaicer.apply(mosaic)
    demosaicer.display(demosaiced)
    
    print("Reconstructing hyperspectral image...")
    reconstructed_hsi = None
    
    print("Evaluating image quality...")
    ssim = None
    psnr = None
    
    


if __name__ == '__main__':
    config = ...
    main(config)
    