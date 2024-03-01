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

path = './CelebA/img_align_celeba/'
dims = 2048
batch_size = 50
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)
path = pathlib.Path(path)
files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in path.glob('*.{}'.format(ext))])

print(f'{len(files)} files in total.')

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
'''
fid_norms = []
c1_norms = []
c2_norms = []


zero_imgs = torch.zeros((sample_size, 3, 299, 299))
zero_act = get_activations(zero_imgs, model, batch_size, dims, device, num_workers, None, raw=True)
zero_mu = np.mean(zero_act, axis=0)
zero_sigma = np.cov(zero_act, rowvar=False)
zero_act = torch.from_numpy(zero_act).to(device, dtype=torch.float32)'''

for strength in strengths:
    if strength == 0:
        transforms=TF.ToTensor()
    else:
        if distortion_type in ['B', 'R', 'E']:
            transforms=TF.Compose([TF.ToTensor(), distortion(strength)])
        else:
            transforms=TF.Compose([TF.ToTensor(), distortion(strength/100)])
            
    print('Distortion type: '+distortion_type+'. Strength: '+str(strength)+'/100.')
    org_files = random.sample(files, sample_size) # random sample_size original images
    original_activations = get_activations(org_files, model, batch_size, dims, device, num_workers, TF.ToTensor())
    original_mu = np.mean(original_activations, axis=0)
    original_sigma = np.cov(original_activations, rowvar=False)
    original_activations = torch.from_numpy(original_activations).to(device, dtype=torch.float32)
    print('Calculated original activations and statistics.')
    
    dist_files = random.sample(files, sample_size) # random sample_size images to distort
    act = get_activations(dist_files, model, batch_size, dims, device, num_workers, transforms)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    act = torch.from_numpy(act).to(device, dtype=torch.float32)
    print('Calculated distorted activations and statistics.')

    fid = calculate_frechet_distance(original_mu, original_sigma, mu, sigma)
    plot_fid.append(fid)
    print('FID:', fid)
    
    #get second sample
    org_files2 = random.sample(files, sample_size) # random sample_size original images
    original_activations2 = get_activations(org_files2, model, batch_size, dims, device, num_workers, TF.ToTensor())
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
    '''
    data_xx, data_yy, data_xy = prepare_1d_samples(activations_x1 = original_activations, 
                                                   activations_x2 = original_activations2, 
                                                   activations_y1 = zero_act, 
                                                   activations_y2 = zero_act, 
                                                   device=device)
    
    fid_norms.append(calculate_frechet_distance(original_mu, original_sigma, zero_mu, zero_sigma))
    c1_norms.append(cramer_iid_score(1, data_xx, data_yy, data_xy, include_third_term=True, device=device))
    c2_norms.append(cramer_iid_score(2, data_xx, data_yy, data_xy, include_third_term=True, device=device))
    print('Calculated norms.')'''


np.save('v1_plot_'+distortion_type+'_fid.npy', np.array(plot_fid))
np.save('v1_plot_'+distortion_type+'_c1.npy', np.array(plot_c1))
np.save('v1_plot_'+distortion_type+'_c2.npy', np.array(plot_c2))
'''
np.save('v1_norms_'+distortion_type+'_fid.npy', np.array(fid_norms))
np.save('v1_norms_'+distortion_type+'_c1.npy', np.array(c1_norms))
np.save('v1_norms_'+distortion_type+'_c2.npy', np.array(c2_norms))'''


print('Saved plots.')