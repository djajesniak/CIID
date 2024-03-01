import torch
from torchvision.transforms.v2 import GaussianBlur, ElasticTransform

def _salt_and_pepper(image, strength):
    """
    Generate salt-and-pepper noise with probability `strength`
    strength: between 0 and 1
    """
    noisy_image = image.clone().detach()
    
    mask = torch.rand(image.shape[1:])
    noisy_image[:,mask < strength/2] = 0
    noisy_image[:,mask > 1 - strength/2] = 1
        
    return noisy_image

def salt_and_pepper(strength):
    return lambda image: _salt_and_pepper(image, strength)

def _gaussian_noise(image, strength):
    noisy_image = image.clone().detach()
    
    # Generate Gaussian noise with mean 0 and standard deviation `strength`
    noise = torch.normal(0, 3*strength, image.shape).to(dtype=torch.float32)
    noisy_image = torch.clip(image + noise, 0, 1).to(dtype=torch.float32)
        
    return noisy_image

def gaussian_noise(strength):
    return lambda image: _gaussian_noise(image, strength)

def gaussian_blur(strength):
    return GaussianBlur(kernel_size=(strength*3, strength*3), sigma=(strength*1.0, strength*1.0))

def elstic_transform(strength):
    return ElasticTransform(alpha=strength*10.)

def _black_rectangles(image, num_rectangles, max_rect_size=None):
    # Copy the original image to avoid modifying the input
    noisy_image = image.clone().detach()

    # Get the image dimensions
    _, height, width = noisy_image.shape
    
    if max_rect_size == None:
        max_rect_size = min(height, width)//6
    
    for _ in range(num_rectangles):
        # Generate random coordinates for the top-left corner of the rectangle
        x = torch.randint(low=0, high=width - max_rect_size, size=(1,1)).item()
        y = torch.randint(low=0, high=height - max_rect_size, size=(1,1)).item()

        # Generate random width and height for the rectangle
        rect_width = torch.randint(low=max_rect_size // 4, high=max_rect_size, size=(1,1)).item()
        rect_height = torch.randint(low=max_rect_size // 4, high=max_rect_size, size=(1,1)).item()

        # Draw a black rectangle on the image
        noisy_image[:,y:y+rect_height, x:x+rect_width] = 0

    return noisy_image

def black_rectangles(num_rectangles, max_rect_size=None):
    return lambda image: _black_rectangles(image, num_rectangles, max_rect_size)