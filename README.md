# NeRF for All: (EECS 504 Project)



```
conda create -n nerf -y python=3.10
conda activate nerf
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchdata==0.6.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

To train the nerf:

```
python nerf.py
```
