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
    if not os.path.exists(config.output_dir):     
        os.makedirs(config.output_dir)
        
    for scene in os.listdir(config.data_dir):
        for pattern in config.pattern:
            for demosaic_method in config.demosaic_method:
                print(f"Scene: {scene} ====== Pattern: {pattern} ====== Demosaic method: {demosaic_method}")
                save_folder = os.path.join(config.output_dir, scene, pattern, demosaic_method)
                process_image(scene, pattern, demosaic_method, save_folder, config)
                print("=====================================================")
        


def process_image(scene, pattern, demosaic_method, save_folder, config):     
    data_processor = HyperspectralImageProcessor(config.output_dir)
    sfa = SFA(pattern)
    demosaicer = Demosaicing(pattern, demosaic_method)
    quality_evaluator = QualityMetrics()
    spectral_evaluator = SpectralEvaluator()
    print("=====================================================")

    print("Loading hyperspectral image... and RGB image")
    ms, rgb = data_processor.load_scene(scene)
    
    print("Applying CFA...")
    mosaic_dict, channel_masks = sfa.apply(ms)
    
        # Mosaic image   
    prev_mosaic = np.zeros_like(rgb.shape[:2])
    for channel in mosaic_dict:
        prev_mosaic += mosaic_dict[channel]
    
    save_image(prev_mosaic, filename=f"mosaic_{pattern}", directory=save_folder, format="png")
    
    save_image(mosaic_dict["R"], filename=f"mosaic_red_{pattern}", directory=save_folder, format="png")
    save_image(mosaic_dict["G"], filename=f"mosaic_green_{pattern}", directory=save_folder, format="png")
    save_image(mosaic_dict["B"], filename=f"mosaic_blue_{pattern}", directory=save_folder, format="png")
    print("=====================================================")
    
    
    
    # Correcting the green channel
    if pattern != "RGGB":
        green = mosaic * channel_masks["G"]
        # Green interpolation kernel
        k_x = np.array([
            [1, 0, 1], 
            [0, 4, 0], 
            [1, 0, 1]]
        ) / 4
        green = convolve2d(green, k_x, 'same') # convolve(green, k_x) # 
        
        mosaic_dict["G"] = green
        save_image(green, filename=f"mosaic_green_{pattern}_corrected", directory=save_folder, format="png")
        print("Green channel corrected.")    
        # Mosaic image   
        mosaic = np.zeros_like(rgb.shape[:2])
        for channel in mosaic_dict:
            mosaic += mosaic_dict[channel]
        
        save_image(mosaic, filename=f"mosaic_{pattern}_corrected", directory=save_folder, format="png")
        print("new mosaic image saved.")
        print("=====================================================")

     
    
    print("Demosaicing image...")
    demosaiced = demosaicer.apply(mosaic, channel_masks)
    
    rgb_hat = np.stack([demosaiced['R'], demosaiced['G'], demosaiced['B']], axis=-1)
    save_image(rgb_hat, filename=f"Demosaiced_{pattern}_{demosaic_method}", directory=save_folder, format="png")

    # save the demosaiced image
    save_image(rgb_hat, filename=f"demosaiced_{pattern}", directory=save_folder, format="png")
    print("=====================================================")
    
    
    
    print("Reconstructing hyperspectral image...")
    reconstructed_hsi = None
    
    print("Evaluating image quality...")
    rgb_image = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float()
    demosaiced = torch.tensor(rgb_hat).permute(2, 0, 1).unsqueeze(0).float()
    
    quality_data = qualityEvaluator.apply(rgb_image, demosaiced)
    print(quality_data)
    qualityEvaluator.save_data(os.path.join(save_folder, "quality_metrics.csv"))
    print("=====================================================")
    
    
if __name__ == '__main__':
    config = Config()
    main(config)
    