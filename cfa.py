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
            'V': 1, # Assuming the 1st band violet (index 0) corresponds to 380-450nm middle of the band 415nm
            'B': 6, # Assuming the 3rd band blue (index 2) corresponds to 450-485nm middle of the band 467.5nm
            'C': 9, # Assuming the 3rd band cyan (index 2) corresponds to 485-500nm  middle of the band 492.5nm
            'G': 13, # Assuming the 3rd band (index 2) corresponds to 500-565nm middle of the band 532.5nm
            'Y': 17, # Assuming the 3rd band (index 2) corresponds to 565-590nm middle of the band 577.5nm
            'O': 20, # Assuming the 3rd band (index 2) corresponds to 590-625nm middle of the band 607.5nm
            'R': 28, # Assuming the 3rd band (index 2) corresponds to 625-750nm  middle of the band 687.5nm
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
        mosaic = {}
        
        HSI = as_float_array(HSI)
        channel_masks = self.masks(HSI.shape[:2])
        channel_keys = list(channel_masks.keys())
        
        for channel in channel_keys:
            tmp = self.normalize_uint8(HSI[..., self.band_mapping[channel]])
            tmp = tmp * channel_masks[channel]
            mosaic[channel] = tmp
            
        return mosaic, channel_masks

# if __name__ == "__main__":
#     pattern = "RGRGBCBORGRGBYBV"
#     HSI = np.random.rand(512, 512, 31) * (2**16 - 1)
#     sfa = SFA(pattern)
#     mosaic = sfa.apply(HSI)

