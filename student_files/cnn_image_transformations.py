import torch
from torchvision.transforms import v2


def create_training_transformations():
    """
    In this function, you are going to preprocess and augment training data.
    Use torchvision.transforms.v2 to do these transforms and the order of the transformations matter!

    First, convert the original PIL Images to Tensors,
          (Hint): Do not directly use ToTensor() instead use v2.ToImage ,v2.ToDtype, and look at ToTensor documentation
    Second, add random horizontal flip with a probability of .2 (RandomApply is not needed)
    Finally, apply random rotation ranging from -36 degrees (clockwise) to 36 degrees (counter clockwise)
             with a probability of .2 (Look at RandomApply)
    RETURN: torchvision.transforms.v2.Compose object
    """
    # convert original PIL images to tensors
        # v2.ToImage, v2.ToDtype, ToTensor
    # add random horizontal flip with a probability of .2

    # apply random rotation ranging from -36 degrees to 36 degrees with a probability of .2 (RandomApply)

    # Return torchvision.transforms.v2.Compose object

    
    raise NotImplementedError()


def create_testing_transformations():
    """
    In this function, you are going to only preprocess testing data.
    Use torchvision.transforms.v2 to do these transforms and the order of the transformations matter!

    Convert the original PIL Images to Tensors
    (Hint): Do not directly use ToTensor() instead use v2.ToImage ,v2.ToDtype, and look at ToTensor documentation

    RETURN: torchvision.transforms.v2.Compose object
    """
    raise NotImplementedError()
