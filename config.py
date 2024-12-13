from typing import Tuple, List
class Config:
    data_dir: str = "CAVE"
    output_dir: str = "output"
    pattern: List[str] = ["RGYB", "RGGB"] # , "RGGB", "RGRGBCBORGRGBYBV"] # ["RGGB", "BGGR", "GRBG", "GBRG"] # 
    demosaic_method: List[str] = ["bilinear", ] # "malvar2004", "menon2007"]
    green_correction_methods = ["interpolation_based_gradient", "interpolation"]
