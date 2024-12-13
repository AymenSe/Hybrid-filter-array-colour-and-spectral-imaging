from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from colour.hints import ArrayLike, Literal, NDArrayFloat
from colour.utilities import as_float_array, tstack, ones, tsplit
from scipy.ndimage.filters import convolve, convolve1d
# from colour_demosaicing.bayer import masks_CFA_Bayer
from cfa import SFA
class Demosaicing:
    """
    A class to perform Bayer CFA demosaicing using various algorithms.
    """
    def __init__(self, pattern: str, demosaic_method: str):
        self.pattern = pattern
        self.demosaic_method = demosaic_method

    def demosaicing_bilinear(self, mosaic, masks):
        
        H_G = as_float_array([
            [0, 1, 0], 
            [1, 4, 1], 
            [0, 1, 0]]
        ) / 4
        
        H_RBY = as_float_array([
            [1, 2, 1], 
            [2, 4, 2], 
            [1, 2, 1]]
        ) / 4

        demosaiced = {}
        # print(mosaic.keys())
        for channel in mosaic.keys():
            if channel in ["R", "B", "Y"]:
                kernel = H_RBY
                # mosaic_channel = mosaic[channel]
            # if channel == "R":
            #     kernel = H_RBY
            #     mosaic_channel = mosaic['R']    
            # elif channel == "B":
            #     kernel = H_RBY
            #     mosaic_channel = mosaic['B']    
            # elif channel == "Y":
            #     kernel = H_RBY
            #     mosaic_channel = mosaic['Y']
            elif channel == "G":
                kernel = H_G
            
            mosaic_channel = mosaic[channel]
                
            # print(f"Demosaicing {channel} channel...")
            # print(f"mask shape: {mask.shape}")
            # print(mask[120:128, 120:128])
            # print(f"tmp shape: {tmp.shape}")
            # print(tmp[120:128, 120:128])
            demosaiced[channel] = convolve(mosaic_channel, kernel)
            # print(f"Demosaicing {channel} channel done.")
            # print(demosaiced[channel][120:128, 120:128])
                
        return demosaiced
            
       
    def demosaicing_malvar2004(self, CFA: ArrayLike) -> NDArrayFloat:

        CFA = np.squeeze(as_float_array(CFA))
        R_m, G_m, B_m = self.masks(CFA.shape)

        GR_GB = (
            as_float_array(
                [
                    [0.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0, 0.0],
                    [-1.0, 2.0, 4.0, 2.0, -1.0],
                    [0.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, 0.0],
                ]
            )
            / 8
        )

        Rg_RB_Bg_BR = (
            as_float_array(
                [
                    [0.0, 0.0, 0.5, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 0.0],
                    [-1.0, 4.0, 5.0, 4.0, -1.0],
                    [0.0, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.5, 0.0, 0.0],
                ]
            )
            / 8
        )

        Rg_BR_Bg_RB = np.transpose(Rg_RB_Bg_BR)

        Rb_BB_Br_RR = (
            as_float_array(
                [
                    [0.0, 0.0, -1.5, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 2.0, 0.0],
                    [-1.5, 0.0, 6.0, 0.0, -1.5],
                    [0.0, 2.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, -1.5, 0.0, 0.0],
                ]
            )
            / 8
        )

        R = CFA * R_m
        G = CFA * G_m
        B = CFA * B_m

        del G_m

        # print(CFA.shape)
        # exit()
        G = np.where(np.logical_or(R_m == 1, B_m == 1), convolve(CFA, GR_GB), G)

        RBg_RBBR = convolve(CFA, Rg_RB_Bg_BR)
        RBg_BRRB = convolve(CFA, Rg_BR_Bg_RB)
        RBgr_BBRR = convolve(CFA, Rb_BB_Br_RR)

        del GR_GB, Rg_RB_Bg_BR, Rg_BR_Bg_RB, Rb_BB_Br_RR

        # Red rows.
        R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * ones(R.shape)
        # Red columns.
        R_c = np.any(R_m == 1, axis=0)[None] * ones(R.shape)
        # Blue rows.
        B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * ones(B.shape)
        # Blue columns
        B_c = np.any(B_m == 1, axis=0)[None] * ones(B.shape)

        del R_m, B_m

        R = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_RBBR, R)
        R = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_BRRB, R)

        B = np.where(np.logical_and(B_r == 1, R_c == 1), RBg_RBBR, B)
        B = np.where(np.logical_and(R_r == 1, B_c == 1), RBg_BRRB, B)

        R = np.where(np.logical_and(B_r == 1, B_c == 1), RBgr_BBRR, R)
        B = np.where(np.logical_and(R_r == 1, R_c == 1), RBgr_BBRR, B)

        del RBg_RBBR, RBg_BRRB, RBgr_BBRR, R_r, R_c, B_r, B_c

        return tstack([R, G, B])

    def _cnv_h(self, x: ArrayLike, y: ArrayLike) -> NDArrayFloat:
        """Perform horizontal convolution."""

        return convolve1d(x, y, mode="mirror")


    def _cnv_v(self, x: ArrayLike, y: ArrayLike) -> NDArrayFloat:
        """Perform vertical convolution."""

        return convolve1d(x, y, mode="mirror", axis=0)


    def demosaicing_CFA_Bayer_Menon2007(
        self, CFA: ArrayLike,
        pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG", "RGXB", "BGXR", "GRBX", "GBRX"] = "RGGB",
        refining_step: bool = True,
    ):
        
        CFA = np.squeeze(as_float_array(CFA))
        R_m, G_m, B_m = self.masks(CFA.shape)

        h_0 = as_float_array([0.0, 0.5, 0.0, 0.5, 0.0])
        h_1 = as_float_array([-0.25, 0.0, 0.5, 0.0, -0.25])

        R = CFA * R_m
        G = CFA * G_m
        B = CFA * B_m

        G_H = np.where(G_m == 0, self._cnv_h(CFA, h_0) + self._cnv_h(CFA, h_1), G)
        G_V = np.where(G_m == 0, self._cnv_v(CFA, h_0) + self._cnv_v(CFA, h_1), G)

        C_H = np.where(R_m == 1, R - G_H, 0)
        C_H = np.where(B_m == 1, B - G_H, C_H)

        C_V = np.where(R_m == 1, R - G_V, 0)
        C_V = np.where(B_m == 1, B - G_V, C_V)

        D_H = np.abs(C_H - np.pad(C_H, ((0, 0), (0, 2)), mode="reflect")[:, 2:])
        D_V = np.abs(C_V - np.pad(C_V, ((0, 2), (0, 0)), mode="reflect")[2:, :])

        del h_0, h_1, CFA, C_V, C_H

        k = as_float_array(
            [
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 3.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )

        d_H = convolve(D_H, k, mode="constant")
        d_V = convolve(D_V, np.transpose(k), mode="constant")

        del D_H, D_V

        mask = d_V >= d_H
        G = np.where(mask, G_H, G_V)
        M = np.where(mask, 1, 0)

        del d_H, d_V, G_H, G_V

        # Red rows.
        R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * ones(R.shape)
        # Blue rows.
        B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * ones(B.shape)

        k_b = as_float_array([0.5, 0, 0.5])

        R = np.where(
            np.logical_and(G_m == 1, R_r == 1),
            G + self._cnv_h(R, k_b) - self._cnv_h(G, k_b),
            R,
        )

        R = np.where(
            np.logical_and(G_m == 1, B_r == 1) == 1,
            G + self._cnv_v(R, k_b) - self._cnv_v(G, k_b),
            R,
        )

        B = np.where(
            np.logical_and(G_m == 1, B_r == 1),
            G + self._cnv_h(B, k_b) - self._cnv_h(G, k_b),
            B,
        )

        B = np.where(
            np.logical_and(G_m == 1, R_r == 1) == 1,
            G + self._cnv_v(B, k_b) - self._cnv_v(G, k_b),
            B,
        )

        R = np.where(
            np.logical_and(B_r == 1, B_m == 1),
            np.where(
                M == 1,
                B + self._cnv_h(R, k_b) - self._cnv_h(B, k_b),
                B + self._cnv_v(R, k_b) - self._cnv_v(B, k_b),
            ),
            R,
        )

        B = np.where(
            np.logical_and(R_r == 1, R_m == 1),
            np.where(
                M == 1,
                R + self._cnv_h(B, k_b) - self._cnv_h(R, k_b),
                R + self._cnv_v(B, k_b) - self._cnv_v(R, k_b),
            ),
            B,
        )

        RGB = tstack([R, G, B])

        del R, G, B, k_b, R_r, B_r

        if refining_step:
            RGB = self.refining_step_Menon2007(RGB, tstack([R_m, G_m, B_m]), M)

        del M, R_m, G_m, B_m

        return RGB


    # self.demosaicing_CFA_Bayer_DDFAPD = demosaicing_CFA_Bayer_Menon2007


    def refining_step_Menon2007(
        self, RGB: ArrayLike, RGB_m: ArrayLike, M: ArrayLike
    ) -> NDArrayFloat:


        R, G, B = tsplit(RGB)
        R_m, G_m, B_m = tsplit(RGB_m)
        M = as_float_array(M)

        del RGB, RGB_m

        # Updating of the green component.
        R_G = R - G
        B_G = B - G

        FIR = ones(3) / 3

        B_G_m = np.where(
            B_m == 1,
            np.where(M == 1, self._cnv_h(B_G, FIR), self._cnv_v(B_G, FIR)),
            0,
        )
        R_G_m = np.where(
            R_m == 1,
            np.where(M == 1, self._cnv_h(R_G, FIR), self._cnv_v(R_G, FIR)),
            0,
        )

        del B_G, R_G

        G = np.where(R_m == 1, R - R_G_m, G)
        G = np.where(B_m == 1, B - B_G_m, G)

        # Updating of the red and blue components in the green locations.
        # Red rows.
        R_r = np.transpose(np.any(R_m == 1, axis=1)[None]) * ones(R.shape)
        # Red columns.
        R_c = np.any(R_m == 1, axis=0)[None] * ones(R.shape)
        # Blue rows.
        B_r = np.transpose(np.any(B_m == 1, axis=1)[None]) * ones(B.shape)
        # Blue columns.
        B_c = np.any(B_m == 1, axis=0)[None] * ones(B.shape)

        R_G = R - G
        B_G = B - G

        k_b = as_float_array([0.5, 0.0, 0.5])

        R_G_m = np.where(
            np.logical_and(G_m == 1, B_r == 1),
            self._cnv_v(R_G, k_b),
            R_G_m,
        )
        R = np.where(np.logical_and(G_m == 1, B_r == 1), G + R_G_m, R)
        R_G_m = np.where(
            np.logical_and(G_m == 1, B_c == 1),
            self._cnv_h(R_G, k_b),
            R_G_m,
        )
        R = np.where(np.logical_and(G_m == 1, B_c == 1), G + R_G_m, R)

        del B_r, R_G_m, B_c, R_G

        B_G_m = np.where(
            np.logical_and(G_m == 1, R_r == 1),
            self._cnv_v(B_G, k_b),
            B_G_m,
        )
        B = np.where(np.logical_and(G_m == 1, R_r == 1), G + B_G_m, B)
        B_G_m = np.where(
            np.logical_and(G_m == 1, R_c == 1),
            self._cnv_h(B_G, k_b),
            B_G_m,
        )
        B = np.where(np.logical_and(G_m == 1, R_c == 1), G + B_G_m, B)

        del B_G_m, R_r, R_c, G_m, B_G

        # Updating of the red (blue) component in the blue (red) locations.
        R_B = R - B
        R_B_m = np.where(
            B_m == 1,
            np.where(M == 1, self._cnv_h(R_B, FIR), self._cnv_v(R_B, FIR)),
            0,
        )
        R = np.where(B_m == 1, B + R_B_m, R)

        R_B_m = np.where(
            R_m == 1,
            np.where(M == 1, self._cnv_h(R_B, FIR), self._cnv_v(R_B, FIR)),
            0,
        )
        B = np.where(R_m == 1, R - R_B_m, B)

        del R_B, R_B_m, R_m

        return tstack([R, G, B])
    
    
    def apply(self, mosaic, masks):
        if self.demosaic_method == "bilinear":
            return self.demosaicing_bilinear(mosaic, masks)
        elif self.demosaic_method == "malvar2004":
            return self.demosaicing_malvar2004(CFA)
        elif self.demosaic_method == "menon2007":
            return self.demosaicing_CFA_Bayer_Menon2007(CFA)
        else:
            raise ValueError(f"Unsupported demosaicing method: {self.demosaic_method}")

    def normalize_uint8(self, image: np.ndarray) -> np.ndarray:
        return (image / image.max() * 255).astype(np.uint8)