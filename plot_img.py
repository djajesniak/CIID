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

sample_size = 8000

'''
sys.argv[1] - distortion type (choose from {'S': salt_and_pepper, 'N': gaussian_noise, 
                                            'B': gaussian_blur, 'E': elstic_transform, 
                                            'R': black_rectangles})
'''

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

'''original_activations = torch.tensor(np.load('original_activations.npy'), dtype=torch.float16)
original_mu = np.load('original_mu.npy')
original_sigma = np.load('original_sigma.npy')
print('Loaded original activations.')'''

try:
    num_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    # os.sched_getaffinity is not available under Windows, use
    # os.cpu_count instead (which may not return the *available* number
    # of CPUs).
    num_cpus = os.cpu_count()
    
num_workers = min(num_cpus, 8) if num_cpus is not None else 0

main_path = '/shared/sets/datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/'

folders = ['n02086646/', 'n02087046/', 'n02088364/', 'n02090622/',
           'n02091467/', 'n02092339/', 'n02091831/', 'n02091244/',
           'n02092002/', 'n02091635/', 'n02091134/', 'n02091032/',
           'n02093859/', 'n02095570/', 'n02095889/', 'n02096051/',
           'n02096177/', 'n02096294/', 'n02096437/', 'n02096585/',
           'n02097047/', 'n02097130/', 'n02097209/', 'n02097298/',
           'n02097474/', 'n02098105/', 'n02098286/', 'n02098413/', #dogs
           'n13040303/', 'n13044778/', 'n13052670/', 'n13054560/'] #fungi

files = []
for folder in folders:
    path = main_path + folder
    path = pathlib.Path(path)
    files += [file for ext in IMAGE_EXTENSIONS
                   for file in path.glob('*.{}'.format(ext))]

files = sorted(files)
    
dims = 2048
batch_size = 50
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)

distortion_types = {'S': salt_and_pepper, 'N': gaussian_noise, 'B': gaussian_blur, 'E': elstic_transform, 'R': black_rectangles}
distortion_type = str(sys.argv[1])
distortion = distortion_types[distortion_type]

if distortion_type == 'R':
    strengths = [0, 1, 3, 5, 7, 11, 15, 21, 35, 51]
elif distortion_type == 'E':
    strengths = [0, 1, 3, 5, 7, 9, 11, 15]
elif distortion_type == 'B':
    strengths = [0, 1, 3, 5, 7, 11, 15, 21]
else:
    strengths = [0, 1, 3, 5, 7.5, 10, 15, 20, 35, 50]
    
plot_fid = []
plot_c1 = []
plot_c2 = []


for strength in strengths:
    if strength == 0:
        transforms=TF.Compose([TF.ToTensor(), TF.Resize((299,299))])
    else:
        if distortion_type in ['B', 'R', 'E']:
            transforms=TF.Compose([TF.ToTensor(), distortion(strength), TF.Resize((299,299))])
        else:
            transforms=TF.Compose([TF.ToTensor(), distortion(strength/100), TF.Resize((299,299))])
            
    print('Distortion type: '+distortion_type+'. Strength: '+str(strength)+'/100.')
    org_files = random.sample(files, sample_size) # random 10k original images
    original_activations = get_activations(org_files, model, batch_size, dims, device, num_workers, 
                                           TF.Compose([TF.ToTensor(), TF.Resize((299,299))]))
    original_mu = np.mean(original_activations, axis=0)
    original_sigma = np.cov(original_activations, rowvar=False)
    original_activations = torch.from_numpy(original_activations).type(torch.float32).to(device)
    print('Calculated original activations and statistics.')
    
    dist_files = random.sample(files, sample_size) # random 10k images to distort
    act = get_activations(dist_files, model, batch_size, dims, device, num_workers, transforms)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    act = torch.from_numpy(act).type(torch.float32).to(device)
    print('Calculated distorted activations and statistics.')

    fid = calculate_frechet_distance(original_mu, original_sigma, mu, sigma)
    plot_fid.append(fid)
    print('FID:', fid)
    
    #get second sample
    org_files2 = random.sample(files, sample_size) # random sample_size original images
    original_activations2 = get_activations(org_files2, model, batch_size, dims, device, num_workers, 
                                            TF.Compose([TF.ToTensor(), TF.Resize((299,299))]))
    original_activations2 = torch.from_numpy(original_activations2).to(device, dtype=torch.float32)
    dist_files2 = random.sample(files, sample_size) # random sample_size images to distort
    act2 = get_activations(dist_files2, model, batch_size, dims, device, num_workers, transforms)
    act2 = torch.from_numpy(act2).to(device, dtype=torch.float32)
    
    data_xx, data_yy, data_xy = prepare_1d_samples(activations_x1 = original_activations, 
                                                   activations_x2 = original_activations2, 
                                                   activations_y1 = act, 
                                                   activations_y2 = act2, 
                                                   device=device)
    
    print('Calculated 1D samples.')
    cramer1 = cramer_iid_score(1, data_xx, data_yy, data_xy, include_third_term=True, device=device)
    plot_c1.append(cramer1)
    print('Cramer-1 (=Wasserstein-1) IID score:', cramer1)
    cramer2 = cramer_iid_score(2, data_xx, data_yy, data_xy, include_third_term=True, device=device)
    plot_c2.append(cramer2)
    print('Cramer-2 (energy distance) IID score:', cramer2)


np.save('img_randomized_plot_'+distortion_type+'_fid.npy', np.array(plot_fid))
np.save('img_randomized_plot_'+distortion_type+'_c1.npy', np.array(plot_c1))
np.save('img_randomized_plot_'+distortion_type+'_c2.npy', np.array(plot_c2))

print('Saved plots.')