import numpy as np
from typing import Tuple, Dict, List
import glob
import os
from PIL import Image

class HyperspectralImageProcessor:
    def __init__(self, db_path) -> None:
        """
        Initialize the processor with the .hdr file path.
        
        Parameters:
        hdr_file_path (str): Path to the .hdr file of the hyperspectral image.
        """
        self.db_path = db_path
        self.scenes = os.listdir(self.db_path)
        # print(self.scenes)
        
        self.data = {
            scene: {
                "ms": sorted(list(glob.glob(os.path.join(self.db_path, scene, scene, "*.png")))),
                "rgb": os.path.join(self.db_path, scene, scene, f"{scene[:-3]}_RGB.bmp"),
            }
            for scene in self.scenes
        }
        
    def normalize_uint8(self, image: np.ndarray) -> np.ndarray:
        return (image / image.max() * 255).astype(np.uint8)
    
    def load_scene_path(self, scene):
        return self.data[scene]["ms"], self.data[scene]["rgb"]
    
    def load_scene(self, scene) -> Tuple[np.ndarray, np.ndarray]:
        ms_paths, rgb_path = self.load_scene_path(scene)
        ms_list = []
        for path in ms_paths:
            s = np.array(Image.open(path))
            if len(s.shape) == 3:
                s = s[:, :, 0]
            assert s.shape == (512, 512), f"Shape mismatch: {s.shape} and path: {path}"
            # s = self.normalize_uint8(s)
            ms_list.append(s)
        ms = np.stack(ms_list, axis=-1)
        rgb = np.array(Image.open(rgb_path))
        return ms, rgb
    