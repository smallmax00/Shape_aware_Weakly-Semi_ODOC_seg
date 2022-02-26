import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision.transforms import functional
import cv2


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'test':
            with open(self._base_dir + '/ext_test.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File('..' + case, 'r')
        image = h5f['img'][:]
        label = h5f['mask'][:]  # * 255
        contour = h5f['con_gau'][:] * 255
        sdm = h5f['sdm'][:]
        vCDR = h5f['cdr'][()]
        if self.split == "train":
            sample = {'image': image, 'label': label, 'con': contour, 'sdm': sdm}
            sample = self.transform(sample)
            sample["idx"] = idx
            sample['cdr'] = vCDR
        else:
            image = functional.to_tensor(image.astype(np.float32))
            label = functional.to_tensor(label.astype(np.uint8))
            contour = functional.to_tensor(contour.astype(np.uint8))
            sdm = functional.to_tensor(sdm.astype(np.float))
            vCDR = torch.from_numpy(np.array(vCDR))

            sample = {'image': image, 'label': label, 'cdr': vCDR, 'con': contour, 'sdm': sdm}
        return sample, case



def random_rot_flip(image, label, contour):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    contour = np.rot90(contour, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    contour = np.flip(contour, axis=axis).copy()
    return image, label, contour


def random_rotate(image, label, contour):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    contour = ndimage.rotate(contour, angle, order=0, reshape=False)
    return image, label, contour



class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, contour = sample['image'], sample['label'], sample['con']#, sample['sdm']
        if random.random() > 0.5:
            image, label, contour = random_rot_flip(image, label, contour)
        elif random.random() > 0.5:
            image, label, contour = random_rotate(image, label, contour)
        image = functional.to_tensor(
            image.astype(np.float32))
        label = functional.to_tensor(label.astype(np.uint8))

        contour = functional.to_tensor(contour.astype(np.uint8))


        sample = {'image': image, 'label': label, 'con': contour}
        return sample



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size



def iterate_once(iterable):
    return np.random.permutation(iterable)



def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())



def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)




