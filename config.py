from typing import Tuple, List
class Config:
    data_dir: str = "data"
    hsi_path: str = None
    img_path: str = None
    y_idx: int = 62
    output_dir: str = "output"
    pattern: List[str] = ["RGYB"] # , "BGXR", "GRBX", "GBRX"] # ["RGGB", "BGGR", "GRBG", "GBRG"] # 
                          
    demosaic_method: List[str] = ["bilinear"] #, "malvar2004", "menon2007"]
    
