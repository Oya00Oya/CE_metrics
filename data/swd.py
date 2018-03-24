from __future__ import division

import numbers
import os
import os.path
import random
import numpy as np
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1):
        w, h = img1.size
        th, tw = self.size
        if w == tw and h == th:  # ValueError: empty range for randrange() (0,0, 0)
            return img1

        if w == tw:
            x1 = 0
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th))

        elif h == th:
            x1 = random.randint(0, w - tw)
            y1 = 0
            return img1.crop((x1, y1, x1 + tw, y1 + th))

        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            return img1.crop((x1, y1, x1 + tw, y1 + th))


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def resize_by(img, side_min):
    return img.resize((int(img.size[0] / min(img.size) * side_min), int(img.size[1] / min(img.size) * side_min)),
                      Image.BICUBIC)


def make_dataset(dir):
    images = []

    for root, __, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return random.sample(images, 2272)


def color_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.root = root
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fpath = self.imgs[index]
        Cimg = color_loader(fpath)
        Cimg = resize_by(Cimg, 512)
        Cimg = RandomCrop(512)(Cimg)
        if random.random() < 0.5:
            Cimg = Cimg.transpose(Image.FLIP_LEFT_RIGHT)

        return np.array(Cimg).astype('uint8').transpose((2, 0, 1))

    def __len__(self):
        return len(self.imgs)


def CreateDataLoader(droot, batchSize):
    random.seed(2333)

    dataset = ImageFolder(root=droot)

    assert dataset

    return data.DataLoader(dataset, batch_size=batchSize,
                           shuffle=True, num_workers=int(4), drop_last=False)
