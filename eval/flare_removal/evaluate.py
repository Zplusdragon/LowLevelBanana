import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb
from torchvision.transforms import ToTensor
from glob import glob
import lpips
from tqdm import tqdm
import torch
import warnings
import csv  
import os

warnings.filterwarnings("ignore")

datasets = {
    "Flare7Kpp_real": {
        "input": "./Flare_Removal/Flare7Kpp_test/real/input",
        "gt": "./Flare_Removal/Flare7Kpp_test/real/gt",
        "output": "./Flare_Removal/Flare7Kpp_test_NBPro/real"
    },
    "Flare7Kpp_synthetic": {
        "input": "./Flare_Removal/Flare7Kpp_test/synthetic/input",
        "gt": "./Flare_Removal/Flare7Kpp_test/synthetic/gt",
        "output": "./Flare_Removal/Flare7Kpp_test_NBPro/synthetic"
    },
    "FlareReal600_2k": {
        "input": "./Flare_Removal/FlareReal600_2k_val/val_input_2k_bicubic",
        "gt": "./Flare_Removal/FlareReal600_2k_val/val_gt_2k_bicubic",
        "output": "./Flare_Removal/FlareReal600_val_NBPro/2k"
    },
    "FlareReal600_4k": {
        "input": "./Flare_Removal/FlareReal600_val/val_input",
        "gt": "./Flare_Removal/FlareReal600_val/val_gt",
        "output": "./Flare_Removal/FlareReal600_val_NBPro/4k"
    }
}

def compare_lpips(img1, img2, loss_fn_alex, device):
    to_tensor = ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0) * 2.0 - 1.0
    img2_tensor = to_tensor(img2).unsqueeze(0) * 2.0 - 1.0
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    
    with torch.no_grad():
        output_lpips = loss_fn_alex(img1_tensor, img2_tensor)
    return output_lpips.cpu().item()

def calculate_metrics(dataset_name, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    dataset = datasets[dataset_name]
    gt_folder = dataset['gt'] + '/*'
    input_folder = dataset['input'] + '/*'
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))

    assert len(gt_list) == len(input_list), "The number of images must match!"
    n = len(gt_list)

    csv_path = os.path.join(output_dir, f"{dataset_name}_evaluate_result.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['gt_path', 'output_path', 'PSNR', 'SSIM', 'LPIPS'])
        
        ssim, psnr, lpips_val = 0, 0, 0
        for i in tqdm(range(n)):
            img_gt = io.imread(gt_list[i])
            img_input = io.imread(input_list[i])
            
            # Handle channel mismatch
            if len(img_gt.shape) == 3 and img_gt.shape[-1] == 3:  # GT is RGB
                if len(img_input.shape) == 2:  # Input is grayscale
                    img_input = gray2rgb(img_input)
            elif len(img_gt.shape) == 2:  # GT is grayscale
                if len(img_input.shape) == 3 and img_input.shape[-1] == 3:
                    img_gt = gray2rgb(img_gt)

            # Resize spatial dimensions only, preserve channels
            target_shape = img_gt.shape[:2] if img_gt.ndim == 3 else img_gt.shape
            img_input = resize(img_input, target_shape, preserve_range=True, anti_aliasing=True).astype(img_gt.dtype)
            
            # Dynamically compute SSIM window size
            min_dim = min(img_gt.shape[:2])
            win_size = min(7, min_dim) if min_dim >= 3 else min_dim
            if win_size % 2 == 0:
                win_size -= 1
            
            # Calculate metrics
            current_ssim = compare_ssim(
                img_gt, img_input, 
                data_range=255, 
                channel_axis=-1 if img_gt.ndim == 3 else None,
                win_size=max(win_size, 3)
            )
            current_psnr = compare_psnr(img_gt, img_input, data_range=255)
            current_lpips = compare_lpips(img_gt, img_input, loss_fn_alex, device)
            
            # Accumulate for averaging
            ssim += current_ssim
            psnr += current_psnr
            lpips_val += current_lpips
            
            # Write current image results to CSV
            writer.writerow([gt_list[i], input_list[i], f"{current_psnr:.4f}", 
                           f"{current_ssim:.4f}", f"{current_lpips:.4f}"])
    
    # Print averages
    ssim /= n
    psnr /= n
    lpips_val /= n
    print(f"Average PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips_val:.4f}")
    print(f"Detailed results saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Flare7Kpp_real")
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = vars(parser.parse_args())

    calculate_metrics(args['dataset'], args['output_dir'])