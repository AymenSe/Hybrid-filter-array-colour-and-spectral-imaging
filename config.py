from typing import Tuple
class Config:
    hsi_path: str = "database/REFLECTANCE_HS-DATASET_2023-03-23_005.hdr"
    output_dir: str = "output"
    rgb_indices: Tuple[int, int, int] = (29, 19, 9)
    band_indices: Tuple[int, ...] = (29,)
    pattern: str = "RGGB"
