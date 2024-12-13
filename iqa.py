import torch
import piq
import pandas as pd



        
class QualityMetrics:
    def __init__(self):
        self.data = {
            "ssim": None,
            "psnr": None,
            "mse": None,
            "fsim": None,
            "vif": None,
            "ms_ssim": None,
            "vsi": None,
            "gmsd": None,
            "lpips": None,
            "pieapp": None,
            "dists": None
        }
    
    # Structural Similarity Index (SSIM)
    def ssim(self, image1, image2):
        return piq.ssim(image1, image2).item()
    
    # Peak Signal-to-Noise Ratio (PSNR)
    def psnr(self, image1, image2):
        return piq.psnr(image1, image2).item()
    
    # Mean Squared Error (MSE)
    def mse(self, image1, image2):
        return torch.mean((image1 - image2) ** 2).item()
    
    # Feature Similarity Index (FSIM)
    def fsim(self, image1, image2):
        return piq.fsim(image1, image2).item()
    
    # # Visual Information Fidelity (VIF)
    def vif(self, image1, image2):
        return piq.vif_p(image1, image2).item()
    
    # Multi-Scale Structural Similarity (MS-SSIM)
    def ms_ssim(self, image1, image2):
        return piq.multi_scale_ssim(image1, image2).item()
    
    # Visual Saliency-based Index (VSI)
    def vsi(self, image1, image2):
        return piq.vsi(image1, image2).item()
    
    # Gradient Magnitude Similarity Deviation (GMSD)
    def gmsd(self, image1, image2):
        return piq.gmsd(image1, image2).item()
    
    def lpips(self, image1, image2):
        return piq.LPIPS()(image1, image2).item()
    
    def pieapp(self, image1, image2):
        return piq.PieAPP()(image1, image2).item()
    
    def dists(self, image1, image2):
        return piq.DISTS()(image1, image2).item()
    
    # Apply all metrics
    def apply(self, image1, image2):
        if not isinstance(image1, torch.Tensor) or not isinstance(image2, torch.Tensor):
            raise ValueError("Inputs must be PyTorch tensors.")
        
        if image1.shape != image2.shape:
            raise ValueError("Input images must have the same shape.")

        # Normalize images to range [0, 1] if needed
        if image1.max() > 1 or image2.max() > 1:
            image1 = image1 / 255.0
            image2 = image2 / 255.0
        
        self.data["ssim"] = self.ssim(image1, image2)
        self.data["psnr"] = self.psnr(image1, image2)
        self.data["mse"] = self.mse(image1, image2)
        self.data["fsim"] = self.fsim(image1, image2)
        self.data["vif"] = self.vif(image1, image2)
        self.data["ms_ssim"] = self.ms_ssim(image1, image2)
        self.data["vsi"] = self.vsi(image1, image2)
        self.data["gmsd"] = self.gmsd(image1, image2)
        self.data["lpips"] = self.lpips(image1, image2)
        self.data["pieapp"] = self.pieapp(image1, image2)
        self.data["dists"] = self.dists(image1, image2)
        
        return self.data
    
    def save_data(self, filename):
        df = pd.DataFrame(self.data, index=[0])
        df.to_csv(filename, index=False)
        print(f"Quality metrics data saved at: {filename}")

from PIL import Image
import numpy as np

def assess_quality(rgb, rgb_hats, save_folder):
    print("Evaluating image quality...")
    quality_evaluator = QualityMetrics()
    rgb = np.array(Image.open(rgb))
    rgb_image = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float()
    rgb_image = rgb_image[:, :-1, :, :]
    print(rgb_image.shape)
    df = pd.DataFrame()
    # df.columns = ["ssim", "psnr", "mse", "fsim", "vif", "ms_ssim", "vsi", "gmsd", "lpips", "pieapp", "dists"]
    for t_img in rgb_hats:
        t_img = np.array(Image.open(t_img))
        demosaiced = torch.tensor(t_img).permute(2, 0, 1).unsqueeze(0).float()
        demosaiced = demosaiced[:, :-1, :, :]
        print(demosaiced.shape)
        quality_data = quality_evaluator.apply(rgb_image, demosaiced)
        # df = df.append(quality_data, ignore_index=True)
        print(quality_data)
    # df.to_csv(save_folder + "/quality_metrics.csv", index=False)
        



if __name__ == "__main__":
    save_folder = "/home/asekhri/Bureau/Hybrid-filter-array-colour-and-spectral-imaging/iqa_output"
    rggb = "/home/asekhri/Bureau/Hybrid-filter-array-colour-and-spectral-imaging/output/balloons_ms/RGGB/interpolation/bilinear/demosaiced_RGB.png"
    rgb_hats = [
        "/home/asekhri/Bureau/Hybrid-filter-array-colour-and-spectral-imaging/output/balloons_ms/RGYB/interpolation/bilinear/demosaiced_RGB.png",
        "/home/asekhri/Bureau/Hybrid-filter-array-colour-and-spectral-imaging/output/balloons_ms/RGYB/interpolation_based_gradient/bilinear/demosaiced_RGB.png"
    ]
    assess_quality(rggb, rgb_hats, save_folder)