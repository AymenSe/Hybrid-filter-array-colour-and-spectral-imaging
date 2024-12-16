import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

from typing import Tuple, Union, Literal
import numpy as np
from colour.utilities import as_float_array, tsplit, validate_method

class SFA:
    def __init__(self, pattern="RGGB"):
        self.pattern = pattern
        self.band_mapping = {
            'V': 1, # Assuming the 1st band violet (index 0) corresponds to 400-450nm middle of the band 415nm
            'B': 6, # list(range(1,11)), # Assuming the 3rd band blue (index 2) corresponds to 450-490nm middle of the band 467.5nm
            'C': 9, # Assuming the 3rd band cyan (index 2) corresponds to 490-500nm  middle of the band 492.5nm
            'G': 16, # list(range(11,21)), # Assuming the 3rd band (index 2) corresponds to 500-565nm middle of the band 532.5nm
            'Y': 17, # Assuming the 3rd band (index 2) corresponds to 565-590nm middle of the band 577.5nm
            'O': 20, # Assuming the 3rd band (index 2) corresponds to 590-625nm middle of the band 607.5nm
            'R': 26,# list(range(21,31)), # Assuming the 3rd band (index 2) corresponds to 625-750nm  middle of the band 687.5nm
        } 
        
    def masks(self, shape):
        assert len(shape) == 2, "Shape must be a tuple of 2 elements."
        
        channels = {}
        for channel in self.pattern:
            channel = channel.upper()
            if channel not in channels:
                    channels[channel] = np.zeros(shape, dtype=bool)    
        if self.pattern == "RGGB":
            for channel, (y, x) in zip(self.pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
                channels[channel][y::2, x::2] = 1
        elif self.pattern == "RGYB":
            for channel, (y, x) in zip(self.pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
                channels[channel][y::2, x::2] = 1
        elif self.pattern == "RGRGBCBORGRGBYBV":
            # Define an extended grid for the larger pattern
            positions = [(0, 0), (0, 1), (0, 2), (0, 3),
                         (1, 0), (1, 1), (1, 2), (1, 3),
                         (2, 0), (2, 1), (2, 2), (2, 3),
                         (3, 0), (3, 1), (3, 2), (3, 3)]
            
            for channel, (y, x) in zip(self.pattern, positions):
                channels[channel][y::4, x::4] = 1
        else:
            raise ValueError(f"Unsupported pattern: {self.pattern}")
        
        return channels

    def apply(self, HSI):
        # print(f"HSI shape: {HSI.shape}")
        dynamic_range = 2**16 - 1
        rgb_dynamic_range = 2**8 - 1
        # # normalize all the bands between 0 and 1
        HSI = HSI / dynamic_range
        # print(HSI.max(), HSI.min())
        
        mosaic = {}
        HSI = as_float_array(HSI)
        channel_masks = self.masks(HSI.shape[:2])
        channel_keys = list(channel_masks.keys())
        
        for channel in channel_keys:
            mask = channel_masks[channel]
            # if channel in ['R', 'G', 'B']:
            #     # band_channel = rgb[:, :, 0]
            #     band_channel = np.stack([HSI[:, :, i] for i in self.band_mapping[channel]], axis=-1)
            #     band_channel = band_channel.mean(axis=-1)
            # else:
            band_channel = HSI[:, :, self.band_mapping[channel]]
                
            band_channel = band_channel.squeeze()
            # from 0-1 to 0-255
            band_channel = band_channel * rgb_dynamic_range
            # assert band_channel.max() <= rgb_dynamic_range, "The band channel max value {} must be less than or equal to the dynamic range of the RGB image {}.".format(band_channel.max(), rgb_dynamic_range)
            # assert band_channel.min() >= 0, "The band channel min value {} must be greater than or equal to 0.".format(band_channel.min())
            
            assert mask.shape == band_channel.shape, "The mask shape {} must be the same as the image shape {}.".format(mask.shape, band_channel.shape)
            mosaic[channel] = band_channel * mask
            
        return mosaic, channel_masks

if __name__ == "__main__":
    pattern = "RGYB"
    HSI = np.random.rand(512, 512, 31) * (2**16 - 1)
    sfa = SFA(pattern)
    mosaic = sfa.apply(HSI)

