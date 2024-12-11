from typing import Tuple, List
class Config:
    data_dir: str = "CAVE"
    hsi_path: str = None
    img_path: str = None
    output_dir: str = "output"
    pattern: List[str] = ["RGGB", "RGYB", "RGRGBCBORGRGBYBV"] # ["RGGB", "BGGR", "GRBG", "GBRG"] # 
    demosaic_method: List[str] = ["bilinear", "malvar2004", "menon2007"]
    
