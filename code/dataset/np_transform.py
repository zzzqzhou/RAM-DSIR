import cv2
import copy
import random
import numpy as np


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        w, h, _ = img.shape
        padw = self.output_size[0] - w if w < self.output_size[0] else 0
        padh = self.output_size[1] - h if h < self.output_size[1] else 0
        img = np.pad(img, ((0, padw), (0, padh), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, padw), (0, padh)), 'constant')

        # cropping
        w, h, _ = img.shape
        x = random.randint(0, w - self.output_size[0])
        y = random.randint(0, h - self.output_size[1])
        img = img[x:x + self.output_size[0], y:y+self.output_size[1], ...]
        mask = mask[x:x + self.output_size[0], y:y+self.output_size[1]]
        return {'img': img, 'mask': mask}


class CenterCrop(object):
    """
    Center crop the image in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        w, h, _ = img.shape
        padw = self.output_size[0] - w if w < self.output_size[0] else 0
        padh = self.output_size[1] - h if h < self.output_size[1] else 0
        img = np.pad(img, ((0, padw), (0, padh), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, padw), (0, padh)), 'constant')

        # cropping
        w, h = img.size
        x = int(round((w - self.output_size[0]) / 2.))
        y = int(round((h - self.output_size[1]) / 2.))
        img = img[x:x + self.output_size[0], y:y+self.output_size[1], ...]
        mask = mask[x:x + self.output_size[0], y:y+self.output_size[1]]
        return {'img': img, 'mask': mask}


class Hflip(object):
    """
    Flip the sample horizontally with p probability
    Args:
    p (float) (0 <= p <= 1): Probability to flip the sample
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        if random.random() < self.p:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return {'img': img, 'mask': mask}


class Random_Resize(object):
    def __init__(self, base_long_size=None, scale_range=(0.75, 1.20)):
        self.base_long_size = base_long_size
        self.scale_range = scale_range
    
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        w, h, _ = img.shape
        if self.base_long_size is None:
            origin_size = h if w > h else w
        else:
            origin_size = self.base_long_size
        long_size = random.randint(int(origin_size * self.scale_range[0]), int(origin_size * self.scale_range[1]))

        if w < h:
            oh = long_size
            ratio = oh / h
            ow = int(w * ratio)
        else:
            ow = long_size
            ratio = ow / w
            oh = int(h * ratio)

        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        return {'img': img, 'mask': mask}


class RandomScaleCrop(object):
    def __init__(self, size):
        self.size = size
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['img']
        mask = sample['mask']
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]

        seed = random.random()
        if seed > 0.5:
            w = int(random.uniform(1, 1.5) * img.shape[0])
            h = int(random.uniform(1, 1.5) * img.shape[1])

            img, mask = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR), cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            sample['img'] = img
            sample['mask'] = mask

        return self.crop(sample)

class GetPair(object):
    def __init__(self, inpaint_rate=0.8):
        self.inpaint_rate = inpaint_rate

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        img_aug = copy.deepcopy(img)

        if random.random() < self.inpaint_rate:
            # Inpainting
            img_aug = image_in_painting(img_aug)
        else:
            # Outpainting
            img_aug = image_out_painting(img_aug)
        return {'img': img, 'img_aug': img_aug, 'mask': mask}

class CreateOnehotLabel(object):
    """ Create Onehot label """

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        img = sample['img']
        mask = sample['mask']
        if 'img_aug' in sample:
            img_aug = sample['img_aug']
        onehot_label = np.zeros((self.num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :] = (mask == i).astype(np.float32)
        if 'img_aug' in sample:
            return {'img': img, 'img_aug': img_aug, 'mask': mask, 'onehot_label': onehot_label}    
        return {'img': img, 'mask': mask, 'onehot_label': onehot_label}


def image_in_painting(image):
    image_np = image.transpose(2, 0, 1)
    _, image_rows, image_cols = image_np.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(image_rows//6, image_rows//3)
        block_noise_size_y = random.randint(image_cols//6, image_cols//3)
        noise_x = random.randint(3, image_rows-block_noise_size_x-3)
        noise_y = random.randint(3, image_cols-block_noise_size_y-3)
        image_np[:,
                 noise_x:noise_x+block_noise_size_x,
                 noise_y:noise_y+block_noise_size_y] = np.random.rand(block_noise_size_x,
                                                                      block_noise_size_y) * 2.0 - 1.0
        cnt -= 1
    image = image_np.transpose(1, 2, 0)
    return image

def image_in_painting_constant(image):
    image_np = image.transpose(2, 0, 1)
    _, image_rows, image_cols = image_np.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(image_rows//6, image_rows//3)
        block_noise_size_y = random.randint(image_cols//6, image_cols//3)
        noise_x = random.randint(3, image_rows-block_noise_size_x-3)
        noise_y = random.randint(3, image_cols-block_noise_size_y-3)
        image_np[:,
                 noise_x:noise_x+block_noise_size_x,
                 noise_y:noise_y+block_noise_size_y] = 1.0
        cnt -= 1
    image = image_np.transpose(1, 2, 0)
    return image

def image_in_painting_rand_constant(image):
    image_np = image.transpose(2, 0, 1)
    _, image_rows, image_cols = image_np.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(image_rows//6, image_rows//3)
        block_noise_size_y = random.randint(image_cols//6, image_cols//3)
        noise_x = random.randint(3, image_rows-block_noise_size_x-3)
        noise_y = random.randint(3, image_cols-block_noise_size_y-3)
        image_np[:,
                 noise_x:noise_x+block_noise_size_x,
                 noise_y:noise_y+block_noise_size_y] = np.ones((block_noise_size_x,
                                                                block_noise_size_y)) * random.random() * 2.0 - 1.0
        cnt -= 1
    image = image_np.transpose(1, 2, 0)
    return image


def image_out_painting(image):
    image_np = image.transpose(2, 0, 1)
    _, image_rows, image_cols = image_np.shape
    image_temp = copy.deepcopy(image_np)
    image_np = np.random.rand(image_np.shape[0], image_np.shape[1], image_np.shape[2]) * 2.0 - 1.0
    block_noise_size_x = image_rows - random.randint(3*image_rows//7, 4*image_rows//7)
    block_noise_size_y = image_cols - random.randint(3*image_cols//7, 4*image_cols//7)
    noise_x = random.randint(3, image_rows-block_noise_size_x-3)
    noise_y = random.randint(3, image_cols-block_noise_size_y-3)
    image_np[:,
             noise_x:noise_x+block_noise_size_x,
             noise_y:noise_y+block_noise_size_y] = image_temp[:,
                                                              noise_x:noise_x+block_noise_size_x,
                                                              noise_y:noise_y+block_noise_size_y]

    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = image_rows - random.randint(3*image_rows//7, 4*image_rows//7)
        block_noise_size_y = image_cols - random.randint(3*image_cols//7, 4*image_cols//7)
        noise_x = random.randint(3, image_rows-block_noise_size_x-3)
        noise_y = random.randint(3, image_cols-block_noise_size_y-3)
        image_np[:,
                 noise_x:noise_x+block_noise_size_x,
                 noise_y:noise_y+block_noise_size_y] = image_temp[:,
                                                                  noise_x:noise_x+block_noise_size_x,
                                                                  noise_y:noise_y+block_noise_size_y]
        
        cnt -= 1
    image = image_np.transpose(1, 2, 0)
    return image

def image_out_painting_constant(image):
    image_np = image.transpose(2, 0, 1)
    _, image_rows, image_cols = image_np.shape
    image_temp = copy.deepcopy(image_np)
    image_np = np.ones((image_np.shape[0], image_np.shape[1], image_np.shape[2]))
    block_noise_size_x = image_rows - random.randint(3*image_rows//7, 4*image_rows//7)
    block_noise_size_y = image_cols - random.randint(3*image_cols//7, 4*image_cols//7)
    noise_x = random.randint(3, image_rows-block_noise_size_x-3)
    noise_y = random.randint(3, image_cols-block_noise_size_y-3)
    image_np[:,
             noise_x:noise_x+block_noise_size_x,
             noise_y:noise_y+block_noise_size_y] = image_temp[:,
                                                              noise_x:noise_x+block_noise_size_x,
                                                              noise_y:noise_y+block_noise_size_y]

    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = image_rows - random.randint(3*image_rows//7, 4*image_rows//7)
        block_noise_size_y = image_cols - random.randint(3*image_cols//7, 4*image_cols//7)
        noise_x = random.randint(3, image_rows-block_noise_size_x-3)
        noise_y = random.randint(3, image_cols-block_noise_size_y-3)
        image_np[:,
                 noise_x:noise_x+block_noise_size_x,
                 noise_y:noise_y+block_noise_size_y] = image_temp[:,
                                                                  noise_x:noise_x+block_noise_size_x,
                                                                  noise_y:noise_y+block_noise_size_y]
        
        cnt -= 1
    image = image_np.transpose(1, 2, 0)
    return image

def image_out_painting_rand_constant(image):
    image_np = image.transpose(2, 0, 1)
    _, image_rows, image_cols = image_np.shape
    image_temp = copy.deepcopy(image_np)
    image_np = np.ones((image_np.shape[0], image_np.shape[1], image_np.shape[2])) * random.random() * 2.0 - 1.0
    block_noise_size_x = image_rows - random.randint(3*image_rows//7, 4*image_rows//7)
    block_noise_size_y = image_cols - random.randint(3*image_cols//7, 4*image_cols//7)
    noise_x = random.randint(3, image_rows-block_noise_size_x-3)
    noise_y = random.randint(3, image_cols-block_noise_size_y-3)
    image_np[:,
             noise_x:noise_x+block_noise_size_x,
             noise_y:noise_y+block_noise_size_y] = image_temp[:,
                                                              noise_x:noise_x+block_noise_size_x,
                                                              noise_y:noise_y+block_noise_size_y]

    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = image_rows - random.randint(3*image_rows//7, 4*image_rows//7)
        block_noise_size_y = image_cols - random.randint(3*image_cols//7, 4*image_cols//7)
        noise_x = random.randint(3, image_rows-block_noise_size_x-3)
        noise_y = random.randint(3, image_cols-block_noise_size_y-3)
        image_np[:,
                 noise_x:noise_x+block_noise_size_x,
                 noise_y:noise_y+block_noise_size_y] = image_temp[:,
                                                                  noise_x:noise_x+block_noise_size_x,
                                                                  noise_y:noise_y+block_noise_size_y]
        
        cnt -= 1
    image = image_np.transpose(1, 2, 0)
    return image