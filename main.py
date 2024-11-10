from data import *
from cfa import *
from demosaicing import *
from reconstruction import *
from iqa import *
from utils import *


def main(config):
    print("Loading hyperspectral image...")
    hsi = ...
    
    print("Creating RGB image...")
    rgb = ...
    
    print("Demosaicing RGB image...")
    demosaiced = ...
    
    print("Reconstructing hyperspectral image...")
    reconstructed_hsi = ...
    
    print("Evaluating image quality...")
    ssim = ...
    psnr = ...
    
    


if __name__ == '__main__':
    config = ...
    main(config)
    