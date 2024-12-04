from data import *
from cfa import *
from demosaicing import *
from reconstruction import *
from iqa import *
from utils import *
from config import Config
from iqa import QualityMetrics
import os
import torch
import os
from scipy.signal import convolve2d

import warnings
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from utils import save_image

def main(config):
    root_hsi = config.hsi_path.split("/")[-1].split(".")[0]
    root_save_folder = os.path.join(config.output_dir, root_hsi)
    if not os.path.exists(root_save_folder):     
        os.makedirs(root_save_folder)
    
    for pattern in config.pattern:
        for demosaic_method in config.demosaic_method:
            save_folder = os.path.join(root_save_folder, pattern, demosaic_method)
            print(f"Pattern: {pattern}, Demosaic method: {demosaic_method}")
            process_image(pattern, demosaic_method, save_folder, config)
            print("=====================================================")
       
       
def process_image(pattern, demosaic_method, save_folder, config):     
    # Initialize objects
    processor = HyperspectralImageProcessor(config.hsi_path, config.img_path)
    # cfa = CFA(config.pattern)
    cfa = BayerCFA(pattern)
    # demosaicer = Demosaicing(config)
    demosaicer = Demosaicing(pattern, demosaic_method)
    qualityEvaluator = QualityMetrics()
    print("=====================================================")
    
    # Example usage:
    # print("Loading hyperspectral image...")
    # metadata = processor.get_metadata()
    # print(metadata)
    # exit()
    # wavelengths = processor.get_wavelengths()
    # print("=====================================================")
    
    
    
    print("Creating RGB image...")
    rgb_image = processor.create_rgb_image()
    save_image(rgb_image, filename="RGB", directory=save_folder, format="png")
    normalized_rgb = processor.normalize_uint8(rgb_image)
    print("=====================================================")
    
    
    print("Applying CFA...")
    mosaic = cfa.apply(normalized_rgb)
    red, green, blue = mosaic
    mosaic = cfa.display(mosaic)
    save_image(mosaic, filename=f"Mosaic_{pattern}", directory=save_folder, format="png")
    save_image(red, filename=f"Mosaic_red_{pattern}", directory=save_folder, format="png")
    save_image(green, filename=f"Mosaic_green_{pattern}", directory=save_folder, format="png")
    save_image(blue, filename=f"Mosaic_blue_{pattern}", directory=save_folder, format="png")
    print("=====================================================")
    
    # print(mosaic.shape)
    
    
    # Correcting the green channel
    if "X" in pattern:
        prev_green = green.copy()
        # Green interpolation kernel
        k_x = 1 / 4 * np.array([
            [1, 0, 1], 
            [0, 4, 0], 
            [1, 0, 1]])
        green = convolve2d(green, k_x, 'same')
        green = processor.normalize_uint8(green)
        
        new_mosaic = (red, green, blue)
        mosaic = cfa.display(new_mosaic)    
        save_image(mosaic, filename=f"Mosaic_{pattern}_corrected", directory=save_folder, format="png")
        save_image(green, filename=f"Mosaic_green_{pattern}_corrected", directory=save_folder, format="png")
        print("Green channel corrected.")
        print("=====================================================")
        
        
    print("Demosaicing RGB image...")
    demosaiced = demosaicer.apply(mosaic)
    demosaiced = processor.normalize_uint8(demosaiced)
    # demosaicer.display(demosaiced)

    # save the demosaiced image
    save_image(demosaiced, filename=f"Demosaiced_{pattern}", directory=save_folder, format="png")
    print("=====================================================")
    
    print("Reconstructing hyperspectral image...")
    reconstructed_hsi = None
    
    print("Evaluating image quality...")
    rgb_image = torch.tensor(rgb_image).permute(2, 0, 1).unsqueeze(0).float()
    demosaiced = torch.tensor(demosaiced).permute(2, 0, 1).unsqueeze(0).float()
    
    quality_data = qualityEvaluator.apply(rgb_image, demosaiced)
    print(quality_data)
    qualityEvaluator.save_data(os.path.join(save_folder, "quality_metrics.csv"))
    print("=====================================================")
    
    


if __name__ == '__main__':
    config = Config()
    all_files = os.listdir(config.data_dir)
    all_files = [f for f in all_files if f.endswith(".hdr")]
    for i, f in enumerate(all_files):
        print(f"{i+1}. {f}")
        config.hsi_path = os.path.join(config.data_dir, f)
        main(config)
    