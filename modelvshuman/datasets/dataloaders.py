import torch
from torchvision import transforms
import torchvision.datasets as datasets
from . import info_mappings


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

    Adapted from:
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __init__(self, *args, **kwargs):

        if "info_mapping" in kwargs.keys():
            self.info_mapping = kwargs["info_mapping"]
            del kwargs["info_mapping"]
        else:
            self.info_mapping = info_mappings.ImageNetInfoMapping()

        super(ImageFolderWithPaths, self).__init__(*args, **kwargs)


    def __getitem__(self, index):
        """override the __getitem__ method. This is the method that dataloader calls."""
        # this is what ImageFolder normally returns
        (sample, target) = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]
        _, _, _, new_target = self.info_mapping(path)
        original_tuple = (sample, new_target)

        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


class PytorchLoader(object):
    """Pytorch Data loader"""

    def __call__(self, path, resize, batch_size, num_workers, single_channel, info_mapping=None):
        """
        Data loader for pytorch models
        :param path:
        :param resize:
        :param batch_size:
        :param num_workers:
        :param single_channel: Boolean whether to conv. to 1 channel
        :return:
        """
        transform_list = []
        # add single channel model support
        if single_channel:
            normalize = transforms.Normalize(mean=[0.449],
                                             std=[0.236])
            transform_list += [
                transforms.Grayscale(num_output_channels=1)]
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        if resize:
            transform_list +=[
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        else:
            transform_list +=[
                transforms.ToTensor(),
                normalize,
            ]
        
        transformations = transforms.Compose(transform_list)
        loader = torch.utils.data.DataLoader(
            ImageFolderWithPaths(path, transformations,
                                 info_mapping=info_mapping),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        return loader
