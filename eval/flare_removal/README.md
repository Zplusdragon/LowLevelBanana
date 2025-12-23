# Evaluation Script for Flare Removal

## Description
Evaluates image restoration quality by calculating PSNR, SSIM, and LPIPS metrics between ground truth and processed images.

## Requirements
- Python 3.7+
- Packages: `skimage`, `tqdm`, `lpips`, `torch`, `torchvision`, `numpy`

## Installation
```bash
pip install scikit-image tqdm lpips torch torchvision numpy
```

## Usage

### Basic Command
```bash
python evaluate.py
```

### Custom Options
```bash
python evaluate.py --dataset "Flare7Kpp_real" --output_dir "./eval_results"
```

### Parameters
- `--dataset`: Dataset name (default: `Flare7Kpp_real`)
  - Options: `Flare7Kpp_real`, `Flare7Kpp_synthetic`, `FlareReal600_2k`, `FlareReal600_4k`
- `--output_dir`: Output directory for results (default: `./eval_results`)

## Dataset Structure
Ensure your data follows this structure (consistent with our Hugging Face repository):
```
./Flare_Removal/
├── Flare7Kpp_test/
│   ├── real/
│   │   ├── input/
│   │   └── gt/
│   └── synthetic/
│       ├── input/
│       └── gt/
├── Flare7Kpp_test_NBPro/
│   ├── real/
│   └── synthetic/
├── FlareReal600_val/
│   ├── val_input/
│   └── val_gt/
├── FlareReal600_2k_val/
│   ├── val_input_2k_bicubic/
│   └── val_gt_2k_bicubic/
└── FlareReal600_val_NBPro/
    ├── 2k/
    └── 4k/
```

## Output
- CSV file: `{dataset_name}_evaluate_result.csv`
- Contains per-image metrics and average values
- Saved to specified `--output_dir`

## Notes
- Automatically handles grayscale/RGB channel mismatches
- Dynamic SSIM window size calculation for different image dimensions
- Requires CUDA for LPIPS computation (falls back to CPU if unavailable)