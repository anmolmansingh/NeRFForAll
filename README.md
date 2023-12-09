# NeRF for All: (EECS 504 Project)

An adaptation of a simple Python NeRF architecture (https://github.com/ayaanzhaque/nerf-from-scratch/tree/main) on the Middlebury dataaset (https://vision.middlebury.edu/mview/data/). Data processing and other changes to the nerf architecture, to adapt to the dataset, included.

Setup:

```
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchdata==0.6.1 --index-url https://download.pytorch.org/whl/cu118
!pip install -r nerf-from-scratch/requirements.txt
```

To train the nerf:

```
python nerf.py
```
