from typing import Tuple, List
class Config:
    data_dir: str = "data"
    hsi_path: str = None
    img_path: str = None
    output_dir: str = "output"
    pattern: List[str] = ["RGGB", "BGGR", "GRBG", "GBRG"] # ["RGXB", "BGXR", "GRBX", "GBRX"] #
                          
    demosaic_method: List[str] = ["bilinear", "malvar2004", "menon2007"]
    
