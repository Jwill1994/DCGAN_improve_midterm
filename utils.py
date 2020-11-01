import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from numpy.random import choice
# Directory containing the data.
train_root = 'data/celebA/'
test_root = 'data/test/'
def get_celeba(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.

    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])

    # Create the dataset.
    dataset = dset.ImageFolder(root=train_root, transform=transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader
    
def get_testset(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.

    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(params['imsize']),
        transforms.CenterCrop(params['imsize']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])

    # Create the dataset.
    dataset = dset.ImageFolder(root=test_root, transform=transform)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['bsize'],
        shuffle=True)

    return dataloader

def label_smoothing(y, device, real = True):
    '''
    Smoothing the label values
    1 will be between 0.7 and 1.2
    0 will be between 0 and 0.2
    '''
    if real:
        return y - 0.3 + (torch.rand(y.shape,device=device)*0.5)
    else: 
        return y + (torch.rand(y.shape,device=device)*0.2)
        


def noisy_labelling(y, p_flip, value_range, smooth_label=True):
    '''
    Return a tensor with noisy labels
    p_flip is the % of labels to which we add noise
    value range is for smooth labels we need to know to which range of values to transform (from [0.7 - 1.2] to [0 - 0.2] for ex)
    '''
    n_select = int(p_flip*y.shape[0])
    flip_x = choice([i for i in range (y.shape[0])],size = n_select)
    if smooth_label:
        y[flip_x] = (((value_range[1] - value_range[0])*(y[flip_x] - min(y)))/(max(y) - min(y))) + value_range[0]
        return y
    else: 
        y[flip_x] = 1 - y[flip_x]
        return y
