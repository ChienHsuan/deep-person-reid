from __future__ import division, print_function, absolute_import
import numpy as np
import math
import random
from collections import deque
import torch
from torchvision.transforms import functional as F
from PIL import Image
import cv2
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip,
    RandomRotation, RandomGrayscale
)


class Random2DTranslation(object):
    """Randomly translates the input image with a probability.

    Specifically, given a predefined shape (height, width), the input is first
    resized with a factor of 1.125, leading to (height*1.125, width*1.125), then
    a random crop is performed. Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)
                                    ), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height)
        )
        return croped_img


class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
        mean (list, optional): erasing value.
    """

    def __init__(
        self,
        probability=0.5,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        mean=[0.4914, 0.4822, 0.4465]
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RandomPadding(object):
    """Random padding
    """

    def __init__(self, p=0.5, padding=(0, 10), **kwargs):
        self.p = p
        self.padding_limits = padding

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        rnd_padding = [random.randint(self.padding_limits[0], self.padding_limits[1]) for _ in range(4)]
        rnd_fill = random.randint(0, 255)

        img = F.pad(img, tuple(rnd_padding), fill=rnd_fill, padding_mode='constant')

        return img


class RandomColorJitter(object):
    def __init__(self, p=0.5, brightness=0.2, contrast=0.15, saturation=0, hue=0, **kwargs):
        self.p = p
        self.transform = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        img = self.transform(img)

        return img


class RandomRotate(object):
    def __init__(self, p=0.5, degrees=(-5, 5), **kwargs):
        self.p = p
        self.transform = RandomRotation(degrees)

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        img = self.transform(img)

        return img


class RandomFigures(object):
    """Insert random figure or some figures from the list [line, rectangle, circle]
    with random color and thickness
    """

    def __init__(self, p=0.5, random_color=True, always_single_figure=False,
                 thicknesses=(1, 6), circle_radiuses=(5, 64), figure_prob=0.5,
                 figures=None, **kwargs):
        self.p = p
        self.random_color = random_color
        self.always_single_figure = always_single_figure
        self.thicknesses = thicknesses
        self.circle_radiuses = circle_radiuses
        self.figure_prob = figure_prob

        if figures is None:
            self.figures = [cv2.line, cv2.rectangle, cv2.circle]
        else:
            assert isinstance(figures, (tuple, list))

            self.figures = []
            for figure in figures:
                assert isinstance(figure, str)

                if hasattr(cv2, figure):
                    self.figures.append(getattr(cv2, figure))
                else:
                    raise ValueError('Unknown figure: {}'.format(figure))

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        if self.always_single_figure:
            figure = [self.figures[random.randint(0, len(self.figures) - 1)]]
        else:
            figure = []
            for i in range(len(self.figures)):
                if random.uniform(0, 1) > self.figure_prob:
                    figure.append(self.figures[i])

        cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = cv_image.shape[:2]
        for f in figure:
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (random.randint(0, w), random.randint(0, h))
            color = tuple([random.randint(0, 256) for _ in range(3)]) if self.random_color else (0, 0, 0)
            thickness = random.randint(*self.thicknesses)
            if f != cv2.circle:
                cv_image = f(cv_image, p1, p2, color, thickness)
            else:
                r = random.randint(*self.circle_radiuses)
                cv_image = f(cv_image, p1, r, color, thickness)

        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img)


class RandomGrid(object):
    """Random grid
    """

    def __init__(self, p=0.5, color=-1, grid_size=(24, 64), thickness=(1, 1), angle=(0, 180), **kwargs):
        self.p = p
        self.color = color
        self.grid_size = grid_size
        self.thickness = thickness
        self.angle = angle

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        if self.color == (-1, -1, -1):  # Random color
            color = tuple([random.randint(0, 256) for _ in range(3)])
        else:
            color = self.color

        grid_size = random.randint(*self.grid_size)
        thickness = random.randint(*self.thickness)
        angle = random.randint(*self.angle)

        return self.draw_grid(img, grid_size, color, thickness, angle)

    @staticmethod
    def draw_grid(image, grid_size, color, thickness, angle):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        mask = np.zeros((h * 8, w * 8, 3), dtype='uint8')
        mask_h, mask_w = mask.shape[:2]
        for i in range(0, mask_h, grid_size):
            p1 = (0, i)
            p2 = (mask_w, i + grid_size)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)
        for i in range(0, mask_w, grid_size):
            p1 = (i, 0)
            p2 = (i + grid_size, mask_h)
            mask = cv2.line(mask, p1, p2, (255, 255, 255), thickness)

        center = (mask_w // 2, mask_h // 2)

        if angle > 0:
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            mask = cv2.warpAffine(mask, rot_mat, (mask_w, mask_h), flags=cv2.INTER_LINEAR)

        offset = (random.randint(-16, 16), random.randint(16, 16))
        center = (center[0] + offset[0], center[1] + offset[1])
        mask = mask[center[1] - h // 2: center[1] + h // 2, center[0] - w // 2: center[0] + w // 2, :]
        mask = cv2.resize(mask, (w, h))
        assert img.shape == mask.shape
        img = np.where(mask == 0, img, color).astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)


class ColorAugmentation(object):
    """Randomly alters the intensities of RGB channels.

    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.

    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor(
            [
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ]
        )
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


def build_transforms(
    height,
    width,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_tr = []

    if 'random_grid' in transforms:
        print('+ random grid')
        transform_tr += [RandomGrid(p=0.15, color=(-1, -1, -1))]

    if 'random_figures' in transforms:
        print('+ random figures')
        transform_tr += [RandomFigures(p=0.33)]

    if 'random_padding' in transforms:
        print('+ random padding')
        transform_tr += [RandomPadding(p=0.25)]

    if 'random_crop' in transforms:
        print(
            '+ random crop (enlarge to {}x{} and '
            'crop {}x{})'.format(
                int(round(height * 1.125)), int(round(width * 1.125)), height,
                width
            )
        )
        transform_tr += [Random2DTranslation(height, width)]

    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [Resize((height, width))]

    if 'random_flip' in transforms:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip(p=0.5)]

    if 'color_jitter' in transforms:
        print('+ color jitter')
        transform_tr += [
            RandomColorJitter(p=0.8, brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1)
        ]

    if 'random_gray_scale' in transforms:
        print('+ random gray scale')
        transform_tr += [RandomGrayscale(p=0.1)]

    if 'random_rotate' in transforms:
        print('+ random rotate')
        transform_tr += [RandomRotate(p=0.33)]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensor()]

    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in transforms:
        print('+ random erase')
        transform_tr += [RandomErasing(mean=norm_mean)]

    transform_tr = Compose(transform_tr)

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_tr, transform_te
