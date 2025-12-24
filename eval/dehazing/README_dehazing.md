
## Environment

```shell

conda create -n pytorch python=3.10
conda activate pytorch

# install pytorch
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
# or
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# or
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install ftfy regex tqdm lmdb numpy==1.26.3 opencv-python requests scikit-image scipy

```


## For Test

```shell

python dehazing.py

```

