from iid_score import prepare_1d_samples, cramer_iid_score
from fid_score import get_activations, IMAGE_EXTENSIONS
import torch
import numpy as np
import random

import lmdb
import io
from PIL import Image
from tqdm import tqdm

import os
import sys
import pathlib
import torchvision.transforms as TF
from inception import InceptionV3

#from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
#from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor

from fld.metrics.FLD import FLD
from fld.metrics.FID import FID

from timeit import default_timer as timer

sample_size = 8000

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

try:
    num_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    num_cpus = os.cpu_count()
    
num_workers = min(num_cpus, 8) if num_cpus is not None else 0

### generated data:
main_path_generated = '/shared/results/gmjajesn/SyleGAN2_cats/stylegan2-config-f-psi-0.5/'

folders_generated = ['0'+str(i)+str(j)+'000' for i in range(10) for j in range(10)]

files_generated = []
for folder in folders_generated:
    path = main_path_generated + folder
    path = pathlib.Path(path)
    files_generated += [file for ext in IMAGE_EXTENSIONS for file in path.glob('*.{}'.format(ext))]

n_files_generated = len(files_generated)
print(f'{n_files_generated} generated samples in total.')


### train data:
lmdb_path = '/shared/sets/datasets/lsun_cat/cats/cat'

# Open the LMDB environment
env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

with env.begin() as txn:
    # Get total number of samples (assume keys are numeric and sequential)
    num_total_samples = txn.stat()['entries']
    print(str(num_total_samples)+' samples in LSUN/cats train dataset.')
        
    # Create a cursor to get all keys
    cursor = txn.cursor()

    # Collect all keys into a list
    lmdb_keys = [key for key, _ in tqdm(cursor, total=num_total_samples, desc="Collecting Keys")]

def load_lmdb_to_tensor(lmdb_path, lmdb_keys, num_samples, size_x, size_y):
    # Open the LMDB environment
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    
    images = []
    
    with env.begin() as txn:
        # Randomly sample num_samples keys
        random_keys = random.sample(lmdb_keys, num_samples)
        
        # Fetch images corresponding to the sampled keys
        for key in tqdm(random_keys, desc="Loading Samples"):
            value = txn.get(key)

            if value is None or len(value) == 0:
                print(f"Warning: Empty or missing value for key: {key}")
                continue  # Skip empty or missing values
            
            try:
                # Try to open the image
                img = Image.open(io.BytesIO(value))
                
                # Optionally resize image to match target size
                img = img.resize((size_x, size_y))
                
                # Convert image to NumPy array and normalize to [0, 1]
                img_np = np.array(img)
                img_np = img_np.astype(np.float32) / 255.0
                
                images.append(img_np)
                
            except Exception as e:
                # Log the error and continue
                print(f"Error processing key {key}: {e}")
                continue
    
    # Stack all images into a single array
    dataset_np = np.stack(images)
    
    return dataset_np


# Desired size for the images (size_x, size_y)
size_x, size_y = 256, 256

dims = 2048
batch_size = 50
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)

###
percentages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

num_experiments = len(percentages)
num_repeats = 10

plot_c2 = np.zeros((num_experiments, num_repeats))
times_fun = np.zeros((num_experiments, num_repeats))

transforms=TF.ToTensor()

for j in range(num_experiments):
    num_copied = int(sample_size * percentages[j])
    
    for i in range(num_repeats):
        train_tensor = load_lmdb_to_tensor(lmdb_path, lmdb_keys, sample_size, size_x, size_y)
        train_activations = get_activations(train_tensor, model, batch_size, dims, device, num_workers, raw=True)
        train_activations = torch.from_numpy(train_activations).to(device, dtype=torch.float32)
        
        if num_copied < sample_size:
            gen_files = random.sample(files_generated, sample_size - num_copied) # random generated images
            act = get_activations(gen_files, model, batch_size, dims, device, num_workers, transforms)
        
        if num_copied > 0:
            copied_tensor = load_lmdb_to_tensor(lmdb_path, lmdb_keys, num_copied, size_x, size_y)
            copied_act = get_activations(copied_tensor, model, batch_size, dims, device, num_workers, raw=True)
            if num_copied >= sample_size:
                act = copied_act[np.random.permutation(copied_act.shape[0])]
            else:
                act = np.vstack((act,copied_act))
                act = act[np.random.permutation(act.shape[0])]
        
        act = torch.from_numpy(act).to(device, dtype=torch.float32)
        
        #get second sample
        train_tensor_2 = load_lmdb_to_tensor(lmdb_path, lmdb_keys, sample_size, size_x, size_y) # random sample_size original images
        train_activations_2 = get_activations(train_tensor_2, model, batch_size, dims, device, num_workers, raw=True)
        train_activations_2 = torch.from_numpy(train_activations_2).to(device, dtype=torch.float32)
        
        if num_copied < sample_size:
            gen_files_2 = random.sample(files_generated, sample_size - num_copied) # random generated images
            act_2 = get_activations(gen_files_2, model, batch_size, dims, device, num_workers, transforms)
        
        if num_copied > 0:
            copied_tensor_2 = load_lmdb_to_tensor(lmdb_path, lmdb_keys, num_copied, size_x, size_y)
            copied_act_2 = get_activations(copied_tensor_2, model, batch_size, dims, device, num_workers, raw=True)
            if num_copied >= sample_size:
                act_2 = copied_act_2[np.random.permutation(copied_act_2.shape[0])]
            else:
                act_2 = np.vstack((act_2,copied_act_2))
                act_2 = act_2[np.random.permutation(act_2.shape[0])]
            
        act_2 = torch.from_numpy(act_2).to(device, dtype=torch.float32)

        data_xx, data_yy, data_xy = prepare_1d_samples(activations_x1 = train_activations, 
                                                       activations_x2 = train_activations_2, 
                                                       activations_y1 = act, 
                                                       activations_y2 = act_2, 
                                                       device=device)
        
        print('Calculated 1D samples.')

        start_time_fun = timer()
        cramer2 = cramer_iid_score(2, data_xx, data_yy, data_xy, include_third_term=True, device=device)
        end_time_fun = timer()
        
        times_fun[j,i] = end_time_fun - start_time_fun
        plot_c2[j,i] = cramer2
        
        print('Cramer-2 (energy distance) IID score:', cramer2)
        print()
        print(f'CIID2: Finished repeat {i+1}/{num_repeats} of experiment {percentages[j]}.')
        print()


np.save('StyleGAN2_cat_psi=0.5_inception_sample_8000_reps_10_c2.npy', np.array(plot_c2))
np.save('StyleGAN2_cat_psi=0.5_inception_sample_8000_reps_10_c2_times_fun.npy', np.array(times_fun))

print('Saved results of StyleGAN2_cat_psi=0.5_inception_sample_8000_reps_10_CIID2.')