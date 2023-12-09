# NeRF for All: (EECS 504 Project)

An adaptation of a simple Python NeRF architecture (https://github.com/ayaanzhaque/nerf-from-scratch/tree/main) on the Middlebury dataset (https://vision.middlebury.edu/mview/data/). Data processing and other changes to the nerf architecture, to adapt to the dataset, included.

![](https://github.com/anmolmansingh/NeRFForAll/tree/main/gifs)
## Environment Setup:

```
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchdata==0.6.1 --index-url https://download.pytorch.org/whl/cu118
!pip install -r nerf-from-scratch/requirements.txt
```

Before training the NeRF, you need to make sure the image data is formatted as an .npz.

For this purpose, we provide you dataloader utilities to run.

Instructions:
1. Download the data from the drive link: https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a (for lego); https://vision.middlebury.edu/mview/data/data/dino.zip (for dino) and arrange into appropriate folders
2. Run the following:
```
python dataloaders/lego_dataloader.py --train_img_dir="path/to/train/image/dir" --train_param_dir="path/to/transforms/train.json" --test_img_dir="path/to/test/image/dir" --test_param_dir="path/to/transforms/train.json" --final_dir= "dir/to/store/file.npz"
```

After data preparation, steps to train the nerf:

1. Move the generated .npz file to "data" folder
2. Ensure you're in the NeRFForAll directory
3. Run the following:

```
python nerf.py --data_file="path/to/npz/file" --batch_size=<batch_size> --iters=<iters> --learning_rate=<learning_rate>
```
