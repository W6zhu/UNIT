"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch.utils.data import DataLoader, Dataset
from networks import Vgg16
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageFilelist, ImageFolder
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time
import nibabel as nib
import glob
import torch.nn.functional as F  # Added import for functional operations (for resizing 3D tensors)
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize as F_resize


# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# vgg_preprocess
# get_scheduler
# weights_init

def collate_resize(batch):
    """
    Dynamically resizes both 2D (image) and 3D (NII volume) batches.
    - For 2D: Resizes using torchvision transforms.
    - For 3D: Resizes slices using torchvision functional resize.
    """
    batch_resized = []
    
    for volume in batch:
        # If it's 4D (C, D, H, W), handle it as a 3D NII volume
        if len(volume.shape) == 4:  
            slices = []
            # Iterate through depth dimension and resize each slice
            for i in range(volume.shape[1]):
                slice_2d = volume[:, i, :, :]  # Extract a 2D slice
                slice_resized = F_resize(slice_2d, (128, 128))  # Resize 2D slice
                slices.append(slice_resized)
            # Stack slices back into 3D volume
            batch_resized.append(torch.stack(slices, dim=1))
        else:  # Otherwise, treat it as a 2D image and apply a normal resize
            resize_transform = transforms.Resize((128, 128))
            batch_resized.append(resize_transform(volume))

    # Return resized batch, ensuring it's in the proper format for further processing
    return torch.stack(batch_resized)



# Add this function to load NII files with 3D transforms
def load_nii_volume(file_path):
    """Load and return a 3D NII volume as a tensor, cast to float32."""
    nii_img = nib.load(file_path)
    volume = torch.tensor(nii_img.get_fdata(), dtype=torch.float32)  # Convert to a float32 tensor
    return volume


def slice_nii_volume_to_2d(volume):
    """Slice a 3D NII volume into a list of 2D slices with debugging info."""
    slices = []
    for i in range(volume.shape[2]):  # Assuming shape is (H, W, D)
        slice_2d = volume[:, :, i]  # Extract 2D slice along the depth dimension
        
        # Debugging: Print min/max values of each slice before processing
        print(f"Slice {i} min: {slice_2d.min()}, max: {slice_2d.max()}")
        
        # Skip slices that are empty or have very low variance
        if slice_2d.max() - slice_2d.min() < 1e-5:
            print(f"Skipping slice {i} due to low variance")
            continue
        
        slice_2d = torch.unsqueeze(torch.unsqueeze(slice_2d, 0), 0)  # Add batch and channel dimensions
        slices.append(slice_2d)
    return slices

def nii_dataset_loader(data_root, folder_name):
    """
    Loads NII files from the specified folder, slices them into 2D images.
    If the folder contains non-NII images (e.g., JPEG), they are loaded as-is.
    """
    folder_path = os.path.join(data_root, folder_name)
    nii_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.nii', '.nii.gz'))]
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

    dataset_slices = []
    
    # Handle NII files (3D)
    for nii_file in nii_files:
        volume = load_nii_volume(nii_file)  # Load NII file as a 3D tensor
        slices = slice_nii_volume_to_2d(volume)  # Convert to 2D slices
        dataset_slices.extend(slices)

    # Handle image files (2D)
    for img_file in image_files:
        image = Image.open(img_file)  # Load image
        image_tensor = transforms.ToTensor()(image)  # Convert image to tensor
        dataset_slices.append(image_tensor)  # Add to dataset

    return dataset_slices

def get_all_data_loaders(config):
    """
    Dynamically loads both 2D and 3D datasets from the specified directories.
    Uses NII loading for 3D files and standard image loading for 2D files.
    """
    data_root = config.get("data_root", None)
    if not data_root or not os.path.exists(data_root):
        raise FileNotFoundError(f"The specified data_root path '{data_root}' does not exist. Please check your path.")

    # Load both 2D and 3D datasets from train and test folders
    trainA_slices = nii_dataset_loader(data_root, 'trainA')
    trainB_slices = nii_dataset_loader(data_root, 'trainB')
    testA_slices = nii_dataset_loader(data_root, 'testA')
    testB_slices = nii_dataset_loader(data_root, 'testB')

    # Create data loaders
    train_loader_a = DataLoader(trainA_slices, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    test_loader_a = DataLoader(testA_slices, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    train_loader_b = DataLoader(trainB_slices, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    test_loader_b = DataLoader(testB_slices, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def get_data_loader_list(root, file_list, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFilelist(root, file_list, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def eformat(f, prec):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d"%(mantissa, int(exp))



def save_slice(slice_array, file_name, rows, cols):
    """Save slices from a 3D array into a grid and save as a PNG image."""
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axs = axs.flatten()

    for i, slice_img in enumerate(slice_array):
        # Move the tensor to CPU and convert to NumPy array
        if slice_img.is_cuda:
            slice_img = slice_img.cpu()
        slice_img = slice_img.numpy()

        # Debugging: print slice shape and range
        print(f"Slice {i} shape before processing: {slice_img.shape}")
        print(f"Slice {i} min: {slice_img.min()}, max: {slice_img.max()}")

        # Handle different shapes
        if slice_img.ndim == 4:
            slice_img = slice_img[0]  # Take the first image in the batch
        if slice_img.ndim == 3:
            slice_img = slice_img[0]  # Take the first channel
        if slice_img.ndim == 2:
            pass  # No change needed

        # Normalize slice for visualization
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-5)
        
        axs[i].imshow(slice_img, cmap='gray')
        axs[i].axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def normalize_image(tensor):
    """Normalize a tensor image to [0, 1] range."""
    min_val = tensor.min()
    max_val = tensor.max()
    # If max and min are the same, we avoid division by zero by returning a zero tensor.
    if max_val == min_val:
        return torch.zeros_like(tensor)
    tensor = (tensor - min_val) / (max_val - min_val + 1e-5)  # Normalize to [0, 1]
    return tensor


def __write_images(image_outputs, display_image_num, file_name):
    # Log the function call
    print(f"__write_images called with file_name: {file_name}")

    processed_images = []
    
    for images in image_outputs:
        # Handle 5D tensors with depth=1
        if images.dim() == 5 and images.shape[2] == 1:  # Check if it's a 5D tensor with depth=1
            images = images.squeeze(2)

        # Process 4D tensors (N, C, H, W)
        if images.dim() == 4:
            # Normalize and expand grayscale images to 3 channels if needed
            images = normalize_image(images)
            images = images.expand(-1, 3, -1, -1)  # Expand grayscale images to 3 channels
            processed_images.append(images)
        else:
            print(f"Skipping image with shape {images.shape}, expected 4D tensor.")

    # Concatenate the images if they are properly processed into 4D
    if len(processed_images) > 0:
        image_tensor = torch.cat([img[:display_image_num] for img in processed_images], 0)
        image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=False)
    
        # Add try-except to log errors during saving
        try:
            print(f"Attempting to save images to {file_name}")
            vutils.save_image(image_grid, file_name, nrow=1)  # Save as PNG
            print(f"Image saved successfully at: {file_name}")
        except Exception as e:
            print(f"Error while saving image to {file_name}: {e}")
    else:
        print(f"No valid 4D images to save for file {file_name}.")


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)

    file_a2b = f'{image_directory}/gen_a2b_{postfix}.png'
    file_b2a = f'{image_directory}/gen_b2a_{postfix}.png'

    # Log file paths and the size of images
    print(f"Saving gen_a2b image to: {file_a2b} (image shape: {image_outputs[0].shape})")
    print(f"Saving gen_b2a image to: {file_b2a} (image shape: {image_outputs[n//2].shape})")

    # Try saving images and catch errors
    try:
        __write_images(image_outputs[0:n//2], display_image_num, file_a2b)  # Save gen_a2b
    except Exception as e:
        print(f"Error while saving gen_a2b images: {e}")

    try:
        __write_images(image_outputs[n//2:n], display_image_num, file_b2a)  # Save gen_b2a
    except Exception as e:
        print(f"Error while saving gen_b2a images: {e}")



def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer)
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and
               ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    
    for m in members:
        value = getattr(trainer, m)
        
        # Check if the value is a tensor and extract the scalar value
        if isinstance(value, torch.Tensor):
            value = value.item()  # Convert tensor to Python scalar
        
        # Log the value if it's a scalar (int or float), skip otherwise
        if isinstance(value, (int, float)):
            train_writer.add_scalar(m, value, iterations + 1)



def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if not os.path.exists(dirname):
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if not gen_models:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


import torchvision.models as models

def load_vgg16(model_dir):
    """Load a pre-trained VGG16 model from torchvision"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define the path for saving and loading the model weights
    weights_path = os.path.join(model_dir, 'vgg16.pth')
    
    # Check if the weights are already downloaded
    if not os.path.exists(weights_path):
        print(f"Downloading pre-trained VGG16 model weights to {weights_path}")
        
        # Download the model weights from torchvision
        vgg16 = models.vgg16(pretrained=True)
        
        # Save the model weights to the specified path
        torch.save(vgg16.state_dict(), weights_path)
    else:
        print(f"Loading pre-trained VGG16 model weights from {weights_path}")
    
    # Create a new VGG16 model
    vgg16 = models.vgg16()
    
    # Load the weights into the model
    vgg16.load_state_dict(torch.load(weights_path))
    
    return vgg16


def vgg_preprocess(batch):
    tensortype = type(batch.data)

    # Check if input is 5D (batch, channel, depth, height, width) - for 3D volumes
    if batch.dim() == 5:
        # Assuming the depth is the 3rd dimension, we take 2D slices along the depth axis
        depth_slices = torch.unbind(batch, dim=2)  # Unbind along depth axis, returns a list of 2D slices
        batch_slices = torch.cat(depth_slices, dim=0)  # Stack the slices along the batch dimension
    else:
        batch_slices = batch

    # If input is grayscale (1 channel), replicate it across 3 channels
    if batch_slices.size(1) == 1:
        batch_slices = batch_slices.repeat(1, 3, 1, 1)  # Replicate grayscale channel to simulate RGB

    # Now the input has 3 channels (RGB-like), we proceed with preprocessing
    (r, g, b) = torch.chunk(batch_slices, 3, dim=1)  # Split into R, G, B channels

    # Convert from RGB to BGR (as expected by VGG)
    batch_slices = torch.cat((b, g, r), dim=1)

    # Scale pixel values from [-1, 1] -> [0, 255]
    batch_slices = (batch_slices + 1) * 255 * 0.5

    # Create a mean tensor to subtract from the image (VGG specific mean values for BGR)
    mean = tensortype(batch_slices.data.size()).cuda()
    mean[:, 0, :, :] = 103.939  # For B channel
    mean[:, 1, :, :] = 116.779  # For G channel
    mean[:, 2, :, :] = 123.680  # For R channel

    # Subtract the mean
    batch_slices = batch_slices.sub(Variable(mean))

    return batch_slices



def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented' % hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                raise ValueError("Unsupported initialization: {}".format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def pytorch03_to_pytorch04(state_dict_base):
    def __conversion_core(state_dict_base):
        state_dict = state_dict_base.copy()
        for key, value in state_dict_base.items():
            if key.endswith(('enc.model.0.norm.running_mean',
                             'enc.model.0.norm.running_var',
                             'enc.model.1.norm.running_mean',
                             'enc.model.1.norm.running_var',
                             'enc.model.2.norm.running_mean',
                             'enc.model.2.norm.running_var',
                             'enc.model.3.model.0.model.1.norm.running_mean',
                             'enc.model.3.model.0.model.1.norm.running_var',
                             'enc.model.3.model.0.model.0.norm.running_mean',
                             'enc.model.3.model.0.model.0.norm.running_var',
                             'enc.model.3.model.1.model.1.norm.running_mean',
                             'enc.model.3.model.1.model.1.norm.running_var',
                             'enc.model.3.model.1.model.0.norm.running_mean',
                             'enc.model.3.model.1.model.0.norm.running_var',
                             'enc.model.3.model.2.model.1.norm.running_mean',
                             'enc.model.3.model.2.model.1.norm.running_var',
                             'enc.model.3.model.2.model.0.norm.running_mean',
                             'enc.model.3.model.2.model.0.norm.running_var',
                             'enc.model.3.model.3.model.1.norm.running_mean',
                             'enc.model.3.model.3.model.1.norm.running_var',
                             'enc.model.3.model.3.model.0.norm.running_mean',
                             'enc.model.3.model.3.model.0.norm.running_var',
                             'dec.model.0.model.0.model.1.norm.running_mean',
                             'dec.model.0.model.0.model.1.norm.running_var',
                             'dec.model.0.model.0.model.0.norm.running_mean',
                             'dec.model.0.model.0.model.0.norm.running_var',
                             'dec.model.0.model.1.model.1.norm.running_mean',
                             'dec.model.0.model.1.model.1.norm.running_var',
                             'dec.model.0.model.1.model.0.norm.running_mean',
                             'dec.model.0.model.1.model.0.norm.running_var',
                             'dec.model.0.model.2.model.1.norm.running_mean',
                             'dec.model.0.model.2.model.1.norm.running_var',
                             'dec.model.0.model.2.model.0.norm.running_mean',
                             'dec.model.0.model.2.model.0.norm.running_var',
                             'dec.model.0.model.3.model.1.norm.running_mean',
                             'dec.model.0.model.3.model.1.norm.running_var',
                             'dec.model.0.model.3.model.0.norm.running_mean',
                             'dec.model.0.model.3.model.0.norm.running_var',
                             )):
                del state_dict[key]
        return state_dict
    state_dict = dict()
    state_dict['a'] = __conversion_core(state_dict_base['a'])
    state_dict['b'] = __conversion_core(state_dict_base['b'])
    return state_dict

