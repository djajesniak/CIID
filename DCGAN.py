import argparse
import os
import sys
import pathlib
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from iid_score import prepare_1d_samples, cramer_iid_score
from fid_score import calculate_frechet_distance, get_activations, IMAGE_EXTENSIONS
from inception import InceptionV3

# Set sample size for metrics calculations
sample_size = 8000
run_name = str(sys.argv[1])
print('Run '+run_name)


# Set random seed for reproducibility
#manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
#random.seed(manualSeed)
#torch.manual_seed(manualSeed)
#torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Root directory for dataset
dataroot = "./CelebA/"

# Number of workers for dataloader
try:
    num_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    # os.sched_getaffinity is not available under Windows, use
    # os.cpu_count instead (which may not return the *available* number
    # of CPUs).
    num_cpus = os.cpu_count()
    
workers = min(num_cpus, 8) if num_cpus is not None else 0

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 7

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 2


# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# Plot some training images
#real_batch = next(iter(dataloader))
#plt.figure(figsize=(8,8))
#plt.axis("off")
#plt.title("Training Images")
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
#plt.savefig('training_images.png')

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    
# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
#  to ``mean=0``, ``stdev=0.02``.
netG.apply(weights_init)

# Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-GPU if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the ``weights_init`` function to randomly initialize all weights
# like this: ``to mean=0, stdev=0.2``.
netD.apply(weights_init)

# Print the model
print(netD)

# Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
img_list_rand = []
G_losses = []
D_losses = []
iters = 0



##############################################
path = './CelebA/img_align_celeba/'
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)
path = pathlib.Path(path)
files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in path.glob('*.{}'.format(ext))])

plot_fid = []
plot_c1 = []
plot_c2 = []

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 150 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 1000 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            gen_noise = torch.randn(sample_size, nz, 1, 1, device=device)
            gen_noise2 = torch.randn(sample_size, nz, 1, 1, device=device)
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                fake2 = netG(torch.randn(64, nz, 1, 1, device=device)).detach().cpu()
                gen_images = netG(gen_noise).detach().cpu()
                gen_images2 = netG(gen_noise2).detach().cpu()
                
            org_files = random.sample(files, sample_size) # random sample_size original images
            original_activations = get_activations(org_files, model, batch_size, dims, device, workers, 
                                                   transforms.Compose([transforms.ToTensor(), transforms.Resize((299,299))]))
            original_mu = np.mean(original_activations, axis=0)
            original_sigma = np.cov(original_activations, rowvar=False)
            original_activations = torch.from_numpy(original_activations).to(device, dtype=torch.float32)
            print('Calculated original activations and statistics.')
            
            act = get_activations((gen_images+1)/2, model, batch_size, dims, device, workers, 
                                  transforms.Resize((299,299)), raw=True) # renormalised from [-1,1] to [0,1]
            mu = np.mean(act, axis=0)
            sigma = np.cov(act, rowvar=False)
            act = torch.from_numpy(act).to(device, dtype=torch.float32)
            print('Calculated generated activations and statistics.')
            
            fid = calculate_frechet_distance(original_mu, original_sigma, mu, sigma)
            plot_fid.append(fid)
            print('FID:', fid)
            
            # get second sample
            org_files2 = random.sample(files, sample_size) # random sample_size original images
            original_activations2 = get_activations(org_files2, model, batch_size, dims, device, workers, 
                                                   transforms.Compose([transforms.ToTensor(), transforms.Resize((299,299))]))
            original_activations2 = torch.from_numpy(original_activations2).to(device, dtype=torch.float32)
            act2 = get_activations((gen_images2+1)/2, model, batch_size, dims, device, workers, 
                                  transforms.Resize((299,299)), raw=True) # renormalised from [-1,1] to [0,1]
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
            
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True)) #sample deterministic images
            img_list_rand.append(vutils.make_grid(fake2, padding=2, normalize=True)) #sample random images

        iters += 1

np.save(run_name+'dcgan_plot_fid.npy', np.array(plot_fid))
np.save(run_name+'dcgan_plot_c1.npy', np.array(plot_c1))
np.save(run_name+'dcgan_plot_c2.npy', np.array(plot_c2))

print('Saved plots.')

for j in range(len(img_list)):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(img_list[j],(1,2,0)))
    plt.savefig(f'{run_name}_generated_images_{j}.png')
    
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(img_list_rand[j],(1,2,0)))
    plt.savefig(f'{run_name}_generated_images_rand_{j}.png')