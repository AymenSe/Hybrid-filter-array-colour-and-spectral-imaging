from typing import Tuple, List
class Config:
    data_dir: str = "CAVE"
    output_dir: str = "output"
    pattern: List[str] = ["RGGB", "RGYB"] # , "RGGB", "RGRGBCBORGRGBYBV"] # ["RGGB", "BGGR", "GRBG", "GBRG"] # 
    demosaic_method: List[str] = ["malvar2004", "bilinear", "menon2007"] # , ] # "bilinear", "menon2007"]
    green_correction_methods = ["interpolation", "interpolation_based_gradient"] 
