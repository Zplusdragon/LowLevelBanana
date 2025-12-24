import cv2
import numpy as np

from skimage.color import deltaE_ciede2000

from basicsr.utils.registry import METRIC_REGISTRY

# im1_lab = cv2.cvtColor(im1, cv2.COLOR_BGR2LAB).astype(np.float32)
# im2_lab = cv2.cvtColor(im2, cv2.COLOR_BGR2LAB).astype(np.float32)
# ciede_values = deltaE_ciede2000(im1_lab, im2_lab).mean()


@METRIC_REGISTRY.register()
def calculate_CIEDE(
        img, # lq
        img2, # gt
        **args
):
    b, c, h, w = img.shape
    device = img.device

    ciede_value_list = []

    for b_i in range(b):
        input_img = img[b_i].detach().cpu().numpy().transpose(1,2,0)
        gt = img2[b_i].detach().cpu().numpy().transpose(1,2,0)

        img_lab = cv2.cvtColor(input_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        gt_lab = cv2.cvtColor(gt, cv2.COLOR_RGB2LAB).astype(np.float32)
        ciede_value = deltaE_ciede2000(img_lab, gt_lab).mean()
        ciede_value_list.append(ciede_value)

    avg_ciede = np.mean(ciede_value_list)

    return avg_ciede


if __name__ == "__main__":

    from pathlib import Path
    from tqdm import tqdm

    gt_folder = Path("")
    input_folder = Path("")

    ciede_value_list = []
    gt_path_list = sorted([x for x in gt_folder.glob("*")])
    input_path_list = sorted([x for x in input_folder.glob("*")])

    assert len(gt_path_list) == len(input_path_list)

    for i, (gt_path, input_path) in tqdm(
        enumerate(zip(gt_path_list, input_path_list)), ncols=50, total=len(gt_path_list)
    ):

        gt = cv2.imread(str(gt_path))
        img = cv2.imread(str(input_path))

        gt_lab = cv2.cvtColor(gt, cv2.COLOR_BGR2LAB).astype(np.float32)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        ciede_value = deltaE_ciede2000(img_lab, gt_lab).mean()
        ciede_value_list.append(ciede_value)
    
    avg_ciede = np.mean(ciede_value_list)

    print(f"Average CIEDE_2000: {avg_ciede}")
