# Is Nano Banana Pro a Low-Level Vision All-Rounder? 🍌
<p align="center">
  <img src="assets/abs.png" alt="Teaser Image" width="60%">
<p align="center">
  

**Jialong Zuo, Haoyou Deng, Hanyu Zhou, Jiaxin Zhu, Yicheng Zhang, Yiwei Zhang, Yongxin Yan, Kaixing Huang, Weisen Chen, Yongtai Deng, Rui Jin, Nong Sang, Changxin Gao**

*School of Artificial Intelligence and Automation, Huazhong University of Science and Technology (HUST)*

<a href="https://lowlevelbanana.github.io/"><img src="https://img.shields.io/badge/Project-Page-green"></a>
<a href="https://huggingface.co/datasets/jlongzuo/LowLevelEval"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue"></a>
<a href="https://github.com/zplusdragon/LowLevelBanana"><img src="https://img.shields.io/badge/GitHub-Repo-black"></a>

---

## 📢 Introduction

This repository hosts the official resources for the technical report: **["Is Nano Banana Pro a Low-Level Vision All-Rounder? A Comprehensive Evaluation on 14 Tasks and 40 Datasets."](https://lowlevelbanana.github.io/assets_common/papers/LowLevelBananaEval_Report.pdf)**.

While commercial T2I models like **Nano Banana Pro** excel in creative synthesis, their potential as generalist solvers for traditional low-level vision challenges remains largely underexplored. In this study, we investigate the critical question: **Is Nano Banana Pro a Low-Level Vision All-Rounder?** We conducted a comprehensive **zero-shot evaluation** across **14 distinct low-level tasks** spanning **40 diverse datasets**.

<p align="center">
  <img src="assets/intro.png" alt="Teaser Image" width="100%">
</p>
<p align="center">
  <em>Figure 1: Exemplary zero-shot results of Nano Banana Pro across 14 low-level vision tasks.</em>
</p>

## 🔥 Key Highlights

- **Massive Benchmark:** Evaluated on **14** low-level vision tasks and **40** datasets.
- **Zero-Shot Setting:** Utilized simple textual prompts without any fine-tuning.
- **The Dichotomy Discovery:** We reveal a distinct performance dichotomy:
    - ✅ **Superior Subjective Quality:** Often hallucinates plausible high-frequency details that surpass specialist models.
    - ❌ **Lower Reference-Based Metrics:** Lags behind in PSNR/SSIM due to the inherent stochasticity of generative models.

## 💻 Evaluation Code
We will provide the complete evaluation code for the quantitative results of each task, which is currently being organized. Please stay tuned.


## 📊 Evaluation Results
*Detailed quantitative and qualitative comparisons can be found in our project page and full report.*

Our extensive analysis identifies Nano Banana Pro as a capable **zero-shot contender** for low-level vision tasks. While it struggles to maintain the strict pixel-level consistency required by conventional metrics (PSNR/SSIM), it offers superior visual quality, suggesting a need for new perception-aligned evaluation paradigms.

We have released the evaluation datasets and corresponding inferred results of Nano Banana Pro used in our study on HuggingFace to facilitate future research.

[**Download the Inferred Results on HuggingFace**](https://huggingface.co/datasets/jlongzuo/LowLevelEval)

## 🔗 Citation

If you find this work helpful for your research, please consider citing:

```bibtex
@techreport{zuo2024nanobanana,
  title={Is Nano Banana Pro a Low-Level Vision All-Rounder? A Comprehensive Evaluation on 14 Tasks and 40 Datasets},
  author={Zuo, Jialong and Deng, Haoyou and Zhou, Hanyu and Zhu, Jiaxin and Zhang, Yicheng and Zhang, Yiwei and Yan, Yongxin and Huang, Kaixing and Chen, Weisen and Deng, Yongtai and Jin, Rui and Sang, Nong and Gao, Changxin},
  institution={Huazhong University of Science and Technology},
  year={2024}
}
