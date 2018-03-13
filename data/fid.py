from __future__ import division

import os
import os.path
import random
import numbers
import numpy as np
from scipy.misc import fromimage
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(root):
    images = []

    for _, __, fnames in sorted(os.walk(root)):
        for fname in fnames:
            if is_image_file(fname):
                images.append(fname)
    return images


def color_loader(path):
    return Image.open(path).convert('RGB')


def resize_by(img, side_min):
    return img.resize((int(img.size[0] / min(img.size) * side_min), int(img.size[1] / min(img.size) * side_min)),
                      Image.BICUBIC)


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


class ImageFolder(data.Dataset):
    def __init__(self, root):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.root = root
        self.imgs = imgs
        self.corp = RandomCrop(299)

    def __getitem__(self, index):
        fname = self.imgs[index]
        Simg = color_loader(os.path.join(self.root, fname))
        Simg = resize_by(Simg, 299)
        Simg = self.corp(Simg)
        if random.random() < 0.5:
            Simg = Simg.transpose(Image.FLIP_LEFT_RIGHT)

        return fromimage(Simg).astype(np.float32)

    def __len__(self):
        return len(self.imgs)


def CreateDataLoader(opt):
    random.seed(opt.manualSeed)

    dataset = ImageFolder(root=opt.dataroot)

    assert dataset

    return data.DataLoader(dataset, batch_size=1,
                           shuffle=True, num_workers=2, drop_last=False)
