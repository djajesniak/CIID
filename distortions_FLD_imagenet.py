from iid_score import prepare_1d_samples, cramer_iid_score
from fid_score import calculate_frechet_distance, get_activations, IMAGE_EXTENSIONS
import torch
import numpy as np
import random

import os
import sys
import pathlib
import torchvision.transforms as TF
from inception import InceptionV3
from distortions import salt_and_pepper, gaussian_noise, gaussian_blur, elstic_transform, black_rectangles

from fld.metrics.FLD import FLD

sample_size = 8000

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

try:
    num_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    num_cpus = os.cpu_count()
    
num_workers = min(num_cpus, 8) if num_cpus is not None else 0

main_path = '/shared/sets/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/'

folders = ['n02086646/', 'n02087046/', 'n02088364/', 'n02090622/',
           'n02091467/', 'n02092339/', 'n02091831/', 'n02091244/',
           'n02092002/', 'n02091635/', 'n02091134/', 'n02091032/',
           'n02093859/', 'n02095570/', 'n02095889/', 'n02096051/',
           'n02096177/', 'n02096294/', 'n02096437/', 'n02096585/',
           'n02097047/', 'n02097130/', 'n02097209/', 'n02097298/',
           'n02097474/', 'n02098105/', 'n02098286/', 'n02098413/',
           'n13040303/', 'n13044778/', 'n13052670/', 'n13054560/']

files = []
for folder in folders:
    path = main_path + folder
    path = pathlib.Path(path)
    files += [file for ext in IMAGE_EXTENSIONS
                   for file in path.glob('*.{}'.format(ext))]

random.shuffle(files)
n_files = len(files)
print('Number of files: ', n_files)
train_files = files[:n_files//3]
test_files = files[n_files//3:]
    
dims = 2048
batch_size = 50
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)

distortion_types = {'S': salt_and_pepper, 'N': gaussian_noise, 'B': gaussian_blur, 'E': elstic_transform, 'R': black_rectangles}
distortion_type = str(sys.argv[1])
distortion = distortion_types[distortion_type]

if distortion_type == 'R':
    strengths = [0, 1, 3, 5, 11, 15, 21, 35, 51]
elif distortion_type == 'E':
    strengths = [0, 1, 3, 5, 9, 11, 15]
elif distortion_type == 'B':
    strengths = [0, 1, 3, 5, 11, 15, 21]
else:
    strengths = [0, 2.5, 5, 7.5, 10, 15, 20, 35, 50]
    
plot_fld = np.zeros((len(strengths), 5))

for i, strength in enumerate(strengths):
    if strength == 0:
        transforms=TF.Compose([TF.ToTensor(), TF.Resize((299,299))])
    else:
        if distortion_type in ['B', 'R', 'E']:
            transforms=TF.Compose([TF.ToTensor(), distortion(strength), TF.Resize((299,299))])
        else:
            transforms=TF.Compose([TF.ToTensor(), distortion(strength/100), TF.Resize((299,299))])
            
    print('\n\nImageNet | Distortion type: '+distortion_type+'. Strength: '+str(strength)+'/100.\n\n')
    
    for j in range(5):
        org_files = random.sample(train_files, sample_size) # random 10k train images
        original_activations = get_activations(org_files, model, batch_size, dims, device, num_workers, 
                                           TF.Compose([TF.ToTensor(), TF.Resize((299,299))]))
        original_activations = torch.from_numpy(original_activations).type(torch.float32).to(device)
    
        dist_files = random.sample(train_files, sample_size) # random 10k images to distort
        act = get_activations(dist_files, model, batch_size, dims, device, num_workers, transforms)
        act = torch.from_numpy(act).type(torch.float32).to(device)
    
        t_files = random.sample(files, sample_size) # random 10k 'test' images
        test_activations = get_activations(t_files, model, batch_size, dims, device, num_workers, 
                                           TF.Compose([TF.ToTensor(), TF.Resize((299,299))]))
        test_activations = torch.from_numpy(test_activations).type(torch.float32).to(device)

        fld = FLD().compute_metric(original_activations, test_activations, act)
        plot_fld[i, j] = fld
        print('\n\n')
        print(f'Str: {strength} | Rep: {j}/10 | FLD:', fld)
        print('\n\n')
    
    

np.save('mean_imagenet_inception_sample_8000_distortion_'+distortion_type+'_fld.npy', np.array(plot_fld))

print('Saved plots.')