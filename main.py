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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from utils import save_image

def main(config, save_folder):
    # Initialize objects
    processor = HyperspectralImageProcessor(config.hsi_path)
    cfa = CFA(config.pattern)
    demosaicer = Demosaicing(config)
    qualityEvaluator = QualityMetrics()
    print("=====================================================")
    
    # Example usage:
    print("Loading hyperspectral image...")
    metadata = processor.get_metadata()
    print(metadata)
    # exit()
    wavelengths = processor.get_wavelengths()
    print("=====================================================")
    
    
    print("Creating RGB image...")
    rgb_image = processor.create_rgb_image(config.rgb_indices)
    save_image(rgb_image, filename="RGB", directory=save_folder, format="png")
    normalized_rgb = processor.normalize_uint8(rgb_image)
    # print(normalized_rgb[:, :, 1][:8, :8])
    print("=====================================================")
    
    
    
    print("Applying CFA...")
    mosaic = cfa.apply(normalized_rgb)
    red, green, blue = mosaic
    rgb = cfa.display(mosaic)
    save_image(rgb, filename=f"Mosaic_{config.pattern}", directory=save_folder, format="png")
    save_image(red, filename=f"Mosaic_red_{config.pattern}", directory=save_folder, format="png")
    save_image(green, filename=f"Mosaic_green_{config.pattern}", directory=save_folder, format="png")
    save_image(blue, filename=f"Mosaic_blue_{config.pattern}", directory=save_folder, format="png")
    print("=====================================================")
    
    print("Demosaicing RGB image...")
    demosaiced = demosaicer.apply(mosaic)
    demosaiced = processor.normalize_uint8(demosaiced)
    demosaicer.display(demosaiced)

    # save the demosaiced image
    save_image(demosaiced, filename=f"Demosaiced_{config.pattern}", directory=save_folder, format="png")
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
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    save_folder = os.path.join(config.output_dir, config.pattern, config.demosaic_method)
    if not os.path.exists(save_folder):     
        os.makedirs(save_folder)
    main(config, save_folder)
    