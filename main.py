from data import *
from cfa import *
from demosaicing import *
from reconstruction import *
from iqa import *
from utils import *
from config import Config

import os

def main(config):
    # Initialize objects
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
    # save the demosaiced image
    save_image(demosaiced, filename=f"Demosaiced_{config.pattern}", directory=config.output, format="png")
    
    print("Reconstructing hyperspectral image...")
    reconstructed_hsi = None
    
    print("Evaluating image quality...")
    ssim = None
    psnr = None
    
    


if __name__ == '__main__':
    config = Config()
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)        
    main(config)
    