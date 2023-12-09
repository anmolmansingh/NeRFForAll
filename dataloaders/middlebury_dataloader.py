import numpy as np
import os
from PIL import Image
import random
import json
import argparse
import sys

def load_images_as_dict(directory):
    images_dict = {}

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                images_dict[filename] = img.copy()  # Load and store the image
    original_size = img.size
    return images_dict, original_size

def load_transforms(path):
    c2ws_img = {}
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:  # Skipping the first line as it might not contain camera parameters
        parts = line.split()
        if len(parts) != 22:  # Ensuring the line has the correct number of parameters
            continue

        img_name = parts[0]
        k_values = list(map(float, parts[1:10]))  # Extracting K matrix values
        r_values = list(map(float, parts[10:19])) # Extracting R matrix values
        t_values = list(map(float, parts[19:22])) # Extracting t vector values

        # Forming K, R, and t
        R = np.array(r_values).reshape(3, 3)
        t = np.array(t_values).reshape(3, 1)

        # Concatenating R and t to form [R | t]
        c2ws = np.hstack([R, t])

        c2ws_homo = np.vstack([c2ws, [0, 0, 0, 1]])

        c2ws_img[img_name] = c2ws_homo

    return c2ws_img

def resize_images_to_numpy(images_dict, new_size = (200, 200)):
    resized_images_dict = {}
    for filename, image in images_dict.items():
        resized_image = image.resize(new_size)
        numpy_image = np.array(resized_image)
        resized_images_dict[filename] = numpy_image[:,:,:3]
    return resized_images_dict

def scale_transform(transforms_dict, scale_mat):
    scaled_transforms_dict = {}
    for filename, transform in transforms_dict.items():
        transform = np.array(transform)
        scaled_transforms = scale_mat @ transform
        scaled_transforms_dict[filename] = scaled_transforms
    return scaled_transforms_dict

# random sample train and val, they should be distinct images

def split_data(image_dict, param_dict, train_frac=0.7, val_frac=0.3):
    # Combine the image and parameter dictionaries
    combined_dict = {key: (image_dict[key], param_dict[key]) for key in image_dict}
    # Convert the dictionary to a list of tuples for shuffling
    combined_list = list(combined_dict.items())
    random.shuffle(combined_list)

    # Calculate split indices
    total_items = len(combined_list)
    train_end = int(train_frac * total_items)
    val_end = train_end + int(val_frac * total_items)

    # Split the data
    train_data = combined_list[:train_end]
    val_data = combined_list[train_end:val_end]

    # Separate the data back into images and parameters
    train_images, train_params = zip(*[(item[1][0], item[1][1]) for item in train_data])
    val_images, val_params = zip(*[(item[1][0], item[1][1]) for item in val_data])

    # Convert tuples back to dictionaries
    train_images_dict = dict(zip([item[0] for item in train_data], train_images))
    val_images_dict = dict(zip([item[0] for item in val_data], val_images))

    train_params_dict = dict(zip([item[0] for item in train_data], train_params))
    val_params_dict = dict(zip([item[0] for item in val_data], val_params))

    return train_images_dict, train_params_dict, val_images_dict, val_params_dict


def dicts_to_arrays(images_dict, params_dict):
    keys = sorted(images_dict.keys())
    images_array = np.array([images_dict[key] for key in keys])
    params_array = np.array([params_dict[key] for key in keys])
    return images_array, params_array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load Lego Dataset')
    parser.add_argument('--train_img_dir', type=str, help='Path to the train image director. Images in PNG format')
    parser.add_argument('--train_param_dir', type = str, help='Path to the train camera parameters. Parameters in .txt format')
    parser.add_argument('--test_img_dir', type=str, help='Path to the test image director. Images in PNG format')
    parser.add_argument('--test_param_dir', type = str, help='Path to the test camera parameters. Parameters in .txt format')
    parser.add_argument('--new_size', type = tuple, help= 'New size of imagein a tuple format(m,n)', default=(200,200))
    parser.add_argument('--final_dir', type=str, help = 'Directory to store the .npz file')
    args = parser.parse_args()
    train_img_dir = args.train_img_dir
    train_param_dir = args.train_param_dir
    test_img_dir = args.test_img_dir
    test_param_dir = args.test_param_dir
    new_size = args.new_size
    final_dir = args.final_dir

    # Loads the train test and val images and camera parameters
    if train_img_dir is not None:
        train_images, train_size = load_images_as_dict(train_img_dir)
    else:
        print('Train image directory missing')
        sys.exit(1)

    if train_param_dir is not None:
        train_trans = load_transforms(train_param_dir)
    else:
        print('Train param directory missing')
        sys.exit(1)

    if test_img_dir is not None:
        test_images, test_size = load_images_as_dict(test_img_dir)
    else:
        print('Test image directory missing')
        sys.exit(1)

    if test_param_dir is not None:
        test_trans = load_transforms(test_param_dir)
    else:
        print('Test param directory missing')
        sys.exit(1)
    # Resizes to new size and returns numpy array with new sizes
    train_images, train_trans, val_images, val_trans = split_data(train_images, train_trans)

    resized_train = resize_images_to_numpy(train_images, new_size)
    resized_test = resize_images_to_numpy(test_images, new_size)
    resized_val = resize_images_to_numpy(val_images, new_size)
    # scaling matrix
    scale_x, scale_y = new_size[0]/train_size[0], new_size[1]/train_size[1]

    scale_mat = np.array([[scale_x, 0, 0, 0], [0, scale_y, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Applying the scaling transform
    train_trans_scaled = scale_transform(train_trans, scale_mat)
    val_trans_scaled = scale_transform(val_trans, scale_mat)
    test_trans_scaled = scale_transform(test_trans, scale_mat)

    # Final images
    train_images_final, train_transforms_final = dicts_to_arrays(resized_train,train_trans_scaled)
    val_images_final, val_transforms_final = dicts_to_arrays(resized_val,val_trans_scaled)
    test_images_final, test_transforms_final = dicts_to_arrays(resized_test,test_trans_scaled)
    focal = np.array([1523.15])
    
    np.savez(final_dir,
         images_train=train_images_final[:100],
         c2ws_train=train_transforms_final[:100],
         images_val=val_images_final[:10],
         c2ws_val=val_transforms_final[:10],
         images_test=test_images_final[:60],
         c2ws_test=test_transforms_final[:60],
        focal  = focal)