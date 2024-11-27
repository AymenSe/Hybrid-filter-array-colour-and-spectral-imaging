from typing import Tuple, List
class Config:
    hsi_path: str = "database/REFLECTANCE_HS-DATASET_2023-03-22_012.hdr"
    output_dir: str = "output"
    rgb_indices: Tuple[int, int, int] = (70, 53, 19)
    band_indices: Tuple[int, ...] = (29,)
    pattern: List[str] = ["RGGB", "BGGR", "GRBG", "GBRG", "RGXB", "GBRX", "GRBX", "BGRX"]
    demosaic_method: List[str] = ["bilinear", "malvar2004", "menon2007"]
    
