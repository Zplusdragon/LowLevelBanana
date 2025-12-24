from pathlib import Path
import sys

import cv2


from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


PRED_DIR = Path("")  
GT_DIR = Path("")        
CROP_BORDER = 0                     
TEST_Y_CHANNEL = False            


def canonical_name(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    # OpenCV loads BGR uint8; metrics expect HWC order and will handle dtype
    return img


def main():
    pred_dir = PRED_DIR
    gt_dir = GT_DIR

    pred_files = sorted([p for p in pred_dir.iterdir() if p.is_file()])
    gt_files = sorted([p for p in gt_dir.iterdir() if p.is_file()])

    gt_map = {canonical_name(p): p for p in gt_files}
    pairs = []
    for p in pred_files:
        key = canonical_name(p)
        gt_path = gt_map.get(key)
        if gt_path:
            pairs.append((p, gt_path))
        else:
            print(f"[warn] no GT match for {p.name}")

    if not pairs:
        print("No matched image pairs found.")
        return

    psnr_list, ssim_list = [], []

    for pred_path, gt_path in pairs:
        pred = load_image(pred_path)
        gt = load_image(gt_path)

        if pred.shape != gt.shape:
 
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
            print(f"[info] resized {pred_path.name} from {pred.shape} to {gt.shape}")

        psnr = calculate_psnr(
            pred, gt, crop_border=CROP_BORDER, input_order="HWC", test_y_channel=TEST_Y_CHANNEL
        )
        ssim = calculate_ssim(
            pred, gt, crop_border=CROP_BORDER, input_order="HWC", test_y_channel=TEST_Y_CHANNEL
        )

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    if psnr_list and ssim_list:
        print("-" * 50)
        print(f"Average PSNR: {sum(psnr_list) / len(psnr_list):.4f}")
        print(f"Average SSIM: {sum(ssim_list) / len(ssim_list):.4f}")


if __name__ == "__main__":
    main()
