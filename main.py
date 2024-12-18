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

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from utils import save_image

def main(config):
    if not os.path.exists(config.output_dir):     
        os.makedirs(config.output_dir)
        
    for scene in os.listdir(config.data_dir):
        if scene == "beads_ms" or scene == "oil_painting_ms":
            for pattern in config.pattern:
                if pattern == "RGGB":
                    for demosaic_method in config.demosaic_method:
                        print(f"Scene: {scene} ====== Pattern: {pattern} ====== Demosaic method: {demosaic_method}")
                        save_folder = os.path.join(config.output_dir, scene, pattern, demosaic_method)
                        process_image(scene, pattern, None, demosaic_method, save_folder, config)
                        print("=====================================================")
                else:
                    for green_correction_method in config.green_correction_methods:
                        for demosaic_method in config.demosaic_method:
                            print(f"Scene: {scene} ====== Pattern: {pattern} ====== Demosaic method: {demosaic_method}")
                            save_folder = os.path.join(config.output_dir, scene, pattern, green_correction_method, demosaic_method)
                            process_image(scene, pattern, green_correction_method, demosaic_method, save_folder, config)
                            print("=====================================================")


def process_image(scene, pattern, green_correction_method, demosaic_method, save_folder, config):     
    data_processor = HyperspectralImageProcessor(config.data_dir)
    sfa = SFA(pattern)
    demosaicer = Demosaicing(pattern, demosaic_method)
    
    # spectral_evaluator = Spectral_Evaluator()
    print("=====================================================")


    print("Loading hyperspectral image... and RGB image")
    ms, rgb = data_processor.load_scene(scene)
    # print(f"RGB shape: {rgb.shape} | MS shape: {ms.shape}")
    # print(f"RGB max: {rgb.max()} | RGB min: {rgb.min()}")
    # save_image(rgb, filename="RGB", directory=save_folder, format="png")
    # print("RGB image saved.")
    # print("=====================================================")
    # print("\n\n")
    true_red = rgb[:, :, 0]
    true_green = rgb[:, :, 1]
    true_blue = rgb[:, :, 2]
    
    
    
    print("Applying CFA...")
    mosaic_dict, channel_masks = sfa.apply(ms)
    # mosaic_dict['R'] = true_red * channel_masks['R']
    # mosaic_dict['G'] = true_green * channel_masks['G']
    # mosaic_dict['B'] = true_blue * channel_masks['B']
    # print(mosaic_dict['G'][120:128, 120:128])
    
    for channel in mosaic_dict.keys():
        save_image(mosaic_dict[channel], filename=f"mosaic_{channel}", directory=save_folder, format="png")
        # print(f"mosaic_dict[channel] shape: {tmp_channel.shape}")
        # print(f"mosaic_dict[channel] max: {tmp_channel.max()} | mosaic_dict[channel] min: {tmp_channel.min()}")
    print("=====================================================")
    print("\n\n")
        
    
    
    print("Correcting the green channel...")
    # Correcting the green channel
    if pattern != "RGGB":
        if green_correction_method == "interpolation":
            print("Green interpolation kernel")
            # Green interpolation kernel
            k_x = np.array([
                [1, 0, 1], 
                [0, 4, 0], 
                [1, 0, 1]]
            ) / 4
            mosaic_dict['G'] = convolve(mosaic_dict['G'], k_x) # convolve(green, k_x) #
            mosaic_dict['Y'] = convolve(mosaic_dict['Y'], k_x) # convolve(yellow, k_x) #
            # mosaic_dict['G'] = (mosaic_dict['G'] / mosaic_dict['G'].max() * 255).astype(np.uint8)
            # print(mosaic_dict['G'][120:128, 120:128])
            
            # print(f"mosaic_dict['G']  shape: {mosaic_dict['G'].shape}")
            # print(f"mosaic_dict['G']  max: {mosaic_dict['G'].max()} | mosaic_dict['G']  min: {mosaic_dict['G'].min()}")
            
        elif green_correction_method == "interpolation_based_gradient":
            print("Green interpolation kernel")
            # Green interpolation kernel
            k_x = np.array([
                [1, 0, 1], 
                [0, 4, 0], 
                [1, 0, 1]]
            ) / 4
            
            g_x = np.array([
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, 0, 4, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0]
            ]) / 4
            
            # mosaic_dict['Y'] = gamma_correction(mosaic_dict['Y'])
            
            gradients_of_yellow = convolve(mosaic_dict['Y'], g_x)
            
            avg_green = convolve(mosaic_dict['G'], k_x)
            
            mosaic_dict['G'] = avg_green + 0.25 * gradients_of_yellow

            gradients_of_green = convolve(mosaic_dict['G'], g_x)
            avg_yellow = convolve(mosaic_dict['Y'], k_x)
            
            mosaic_dict['Y'] = avg_yellow + 0.25 * gradients_of_green
            
        save_image(mosaic_dict['G'], filename=f"mosaic_G_corrected", directory=save_folder, format="png")
        save_image(mosaic_dict['Y'], filename=f"mosaic_Y_corrected", directory=save_folder, format="png")
        print("Green channel corrected.")    
        print("=====================================================")
        print("\n\n")

    # print("Creating the mosaic image...")

    # # mosaic = np.stack([mosaic_tmp for mosaic_tmp in mosaic_dict.values()], axis=-1)
    # # print(mosaic.shape)
    # # exit()
    # # mosaic = mosaic.sum(axis=2)
    # # save_image(mosaic, filename=f"mosaic", directory=save_folder, format="png")    
    # # print(f"mosaic  shape: {mosaic.shape}")
    # # print(f"mosaic  max: {mosaic.max()} | mosaic  min: {mosaic.min()}")
    # # print("Mosaic image created.")
    # print("=====================================================")
    # print("\n\n")
    
    
    
    
    print("Demosaicing image...")
    demosaiced = demosaicer.apply(mosaic_dict, channel_masks)
    for channel in demosaiced.keys():
        save_image(demosaiced[channel], filename=f"demosaiced_{channel}", directory=save_folder, format="png")
        # print(f"demosaiced[channel] shape: {demosaiced[channel].shape}")
        # print(f"demosaiced[channel] max: {demosaiced[channel].max()} | demosaiced[channel] min: {demosaiced[channel].min()}")

    # print(f"demosaiced['G']  shape: {demosaiced['G'].shape}")
    # print(f"demosaiced['G']  max: {demosaiced['G'].max()} | demosaiced['G']  min: {demosaiced['G'].min()}")
    # print(demosaiced['G'][120:128, 120:128])
    print("Green channel demosaiced.")
    print("=====================================================")
    print("\n\n")
    
    
    print("Reconstructing RGB image...")
    # print(demosaiced.keys())
    # exit()
    rgb_hat = tstack([demosaiced['R'], demosaiced['G'], demosaiced['B']])
    # print(f"rgb_hat  shape: {rgb_hat.shape}")
    # if green_correction_method == "interpolation_based_gradient":
        # rgb_hat = gamma_correction(rgb_hat, gamma=2.0)
    rgb_hat = gamma_correction(rgb_hat)
    rgb_hat = normalize_uint8(rgb_hat)
    print(rgb_hat.max(), rgb_hat.min(), rgb_hat.mean())
    # print(f"rgb_hat  shape: {rgb_hat.shape}")
    # print(f"rgb_hat  max: {rgb_hat.max()} | rgb_hat min: {rgb_hat.min()}")
    save_image(rgb_hat, filename=f"demosaiced_RGB", directory=save_folder, format="png")
    print("Demosaiced RGB image saved.")
    print("=====================================================")
    
    # exit()
    # print("Reconstructing hyperspectral image...")
    # reconstructed_hsi = None
    
def normalize_uint8(image):
    #normalize image to range [0, 255]
    return (image / image.max() * 255).astype(np.uint8)
        
if __name__ == '__main__':
    config = Config()
    # from PIL import Image

    # # Open the BMP file
    # file_path = 'CAVE/beads_ms/beads_ms/beads_RGB.bmp'  # Replace with your BMP file path
    # import numpy as np

    # # Open BMP and extract pixel data
    # with Image.open(file_path) as img:
    #     pixel_data = np.array(img)  # Convert image to NumPy array
    #     print("Pixel Data Shape:", pixel_data.shape)  # (height, width, channels)

    # # Access specific pixel values
    # print("Pixel at (0, 0):", pixel_data[0, 0])


    # import matplotlib.pyplot as plt

    # # Extract individual color channels
    # r, g, b = pixel_data[:, :, 0], pixel_data[:, :, 1], pixel_data[:, :, 2]

    # # Plot histograms for each channel
    # plt.hist(r.ravel(), bins=256, color='red', alpha=0.5, label='Red')
    # plt.hist(g.ravel(), bins=256, color='green', alpha=0.5, label='Green')
    # plt.hist(b.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
    # plt.legend()
    # plt.title("RGB Color Distribution")
    # plt.show()

        
    # exit()
    main(config)
    