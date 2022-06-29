import copy
import torch
import random
import numpy as np

from scipy import ndimage
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def to_multilabel(pre_mask, classes = 2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [0, 1]
    mask[pre_mask == 2] = [1, 1]
    return mask

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
        w, h = img.size
        padw = self.output_size[0] - w if w < self.output_size[0] else 0
        padh = self.output_size[1] - h if h < self.output_size[1] else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

        # cropping
        w, h = img.size
        x = random.randint(0, w - self.output_size[0])
        y = random.randint(0, h - self.output_size[1])
        img = img.crop((x, y, x + self.output_size[0], y + self.output_size[1]))
        mask = mask.crop((x, y, x + self.output_size[0], y + self.output_size[1]))

        if 'img_freq' in sample.keys():
            img_freq = sample['img_freq'].crop((x, y, x + self.output_size[0], y + self.output_size[1]))
            return {'img': img, 'mask': mask, 'img_freq': img_freq}

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
        w, h = img.size
        padw = self.output_size[0] - w if w < self.output_size[0] else 0
        padh = self.output_size[1] - h if h < self.output_size[1] else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

        # cropping
        w, h = img.size
        x = int(round((w - self.output_size[0]) / 2.))
        y = int(round((h - self.output_size[1]) / 2.))
        img = img.crop((x, y, x + self.output_size[0], y + self.output_size[1]))
        mask = mask.crop((x, y, x + self.output_size[0], y + self.output_size[1]))

        if 'img_freq' in sample.keys():
            img_freq = sample['img_freq'].crop((x, y, x + self.output_size[0], y + self.output_size[1]))
            return {'img': img, 'mask': mask, 'img_freq': img_freq}

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
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
            if 'img_freq' in sample.keys():
                img_freq = sample['img_freq'].transpose(Image.FLIP_LEFT_RIGHT)
                return {'img': img, 'mask': mask, 'img_freq': img_freq}
        
        if 'img_freq' in sample.keys():
            return {'img': img, 'mask': mask, 'img_freq': sample['img_freq']}
        return {'img': img, 'mask': mask}


class Random_Resize(object):
    def __init__(self, base_long_size=None, scale_range=(0.75, 1.20)):
        self.base_long_size = base_long_size
        self.scale_range = scale_range
    
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        w, h = img.size
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

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if 'img_freq' in sample.keys():
            img_freq = sample['img_freq'].resize((ow, oh), Image.BILINEAR)
            return {'img': img, 'mask': mask, 'img_freq': img_freq}

        return {'img': img, 'mask': mask}


class Resize_Ratio(object):
    def __init__(self, base_size, ratio_range):
        self.base_size = base_size
        self.ratio_range = ratio_range

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        w, h = img.size
        long_side = random.randint(int(self.base_size * self.ratio_range[0]),
                                   int(self.base_size * self.ratio_range[1]))
        
        if h > w:
            oh = long_side
            ow = int(1.0 * w * long_side / h + 0.5)
        else:
            ow = long_side
            oh = int(1.0 * h * long_side / w + 0.5)

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if 'img_freq' in sample.keys():
            img_freq = sample['img_freq'].resize((ow, oh), Image.BILINEAR)
            return {'img': img, 'mask': mask, 'img_freq': img_freq}

        return {'img': img, 'mask': mask}


class Resize(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']

        img = img.resize((self.target_size[0], self.target_size[1]), Image.BILINEAR)
        mask = mask.resize((self.target_size[0], self.target_size[1]), Image.NEAREST)

        if 'img_freq' in sample.keys():
            img_freq = sample['img_freq'].resize((self.target_size[0], self.target_size[1]), Image.BILINEAR)
            return {'img': img, 'mask': mask, 'img_freq': img_freq}

        return {'img': img, 'mask': mask}


class RandomScaleCrop(object):
    def __init__(self, size):
        self.size = size
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['img']
        mask = sample['mask']
        assert img.width == mask.width
        assert img.height == mask.height

        seed = random.random()
        if seed > 0.5:
            w = int(random.uniform(1, 1.5) * img.size[0])
            h = int(random.uniform(1, 1.5) * img.size[1])

            img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
            sample['img'] = img
            sample['mask'] = mask

            if 'img_freq' in sample.keys():
                img_freq = sample['img_freq'].resize((w, h), Image.BILINEAR)
                sample['img_freq'] = img_freq

        return self.crop(sample)


class Rotate(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        
        degree = random.randint(-20, 20)
        img = img.rotate(degree, Image.BILINEAR)
        mask = mask.rotate(degree, Image.NEAREST, fillcolor=255)
        if 'img_freq' in sample.keys():
            img_freq = sample['img_freq'].rotate(degree, Image.BILINEAR)
            return {'img': img, 'mask': mask, 'img_freq': img_freq}
        return {'img': img, 'mask': mask}


class Blur(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        if random.random() < self.p:
            sigma = np.random.uniform(0.1, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return {'img': img, 'mask': mask}


class CutOut(object):
    def __init__(self, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
                 ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
        self.p = p
        self.size_min = size_min
        self.size_max = size_max
        self.ratio_1 = ratio_1
        self.ratio_2 = ratio_2
        self.value_min = value_min
        self.value_max = value_max
        self.pixel_level = pixel_level

    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        if random.random() < self.p:
            img = np.array(img)
            mask = np.array(mask)

            img_h, img_w, img_c = img.shape

            while True:
                size = np.random.uniform(self.size_min, self.size_max) * img_h * img_w
                ratio = np.random.uniform(self.ratio_1, self.ratio_2)
                erase_w = int(np.sqrt(size / ratio))
                erase_h = int(np.sqrt(size * ratio))
                x = np.random.randint(0, img_w)
                y = np.random.randint(0, img_h)

                if x + erase_w <= img_w and y + erase_h <= img_h:
                    break
            
            if self.pixel_level:
                value = np.random.uniform(self.value_min, self.value_max, (erase_h, erase_w, img_c))
            else:
                value = np.random.uniform(self.value_min, self.value_max)
            
            img[y:y + erase_h, x:x + erase_w] = value
            mask[y:y + erase_h, x:x + erase_w] = 255

            img = Image.fromarray(img.astype(np.uint8))
            mask = Image.fromarray(mask.astype(np.uint8))

        return {'img': img, 'mask': mask}


class Sharpness(object):
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        if random.random() < self.p:
            v = random.uniform(0.05, 0.95)
            img = ImageEnhance.Sharpness(img).enhance(v)
        return {'img': img, 'mask': mask}


class Solarize(object):
    def __init__(self, p=0.2):
        self.p = p
    
    def __call__(self, sample):
        img, mask = sample['img'], sample['mask']
        if random.random() < self.p:
            threshold = random.randint(0, 256)
            img = ImageOps.solarize(img, threshold=threshold)
        return {'img': img, 'mask': mask}

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

class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        img = np.array(sample['img']).astype(np.float32)
        __mask = np.array(sample['mask']).astype(np.uint8)

        img /= 127.5
        img -= 1.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        if 'img_aug' in sample:
            img_aug = np.array(sample['img_aug']).astype(np.float32)
            img_aug /= 127.5
            img_aug -= 1.0
            img_aug = img_aug.transpose(2, 0, 1)
            img_aug = torch.from_numpy(img_aug).float()
        
        if 'img_freq' in sample:
            img_freq = np.array(sample['img_freq']).astype(np.float32)
            img_freq /= 127.5
            img_freq -= 1.0
            img_freq = img_freq.transpose(2, 0, 1)
            img_freq = torch.from_numpy(img_freq).float()

        if __mask is not None:
            _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
            _mask[__mask > 200] = 255
            _mask[(__mask > 50) & (__mask < 201)] = 128
            _mask[(__mask > 50) & (__mask < 201)] = 128

            __mask[_mask == 0] = 2
            __mask[_mask == 255] = 0
            __mask[_mask == 128] = 1

            mask = to_multilabel(__mask)
            mask = mask.transpose(2, 0, 1)
            mask = torch.from_numpy(np.array(mask)).float()

            if 'img_aug' in sample:
                if 'img_freq' in sample:
                    return {'img': img, 'img_aug': img_aug, 'mask': mask, 'img_freq': img_freq}
                return {'img': img, 'img_aug': img_aug, 'mask': mask}
            
            if 'img_freq' in sample:
                return {'img': img, 'mask': mask, 'img_freq': img_freq}
            return {'img': img, 'mask': mask}

        if 'img_aug' in sample:
            if 'img_freq' in sample:
                return {'img': img, 'img_aug': img_aug, 'img_freq': img_freq}
            return {'img': img, 'img_aug': img_aug}
        
        if 'img_freq' in sample:
            return {'img': img, 'img_freq': img_freq}
        return {'img': img}

class GetBoundary(object):
    def __init__(self, width = 5):
        self.width = width
    def __call__(self, mask):
        cup = mask[:, :, 0]
        disc = mask[:, :, 1]
        dila_cup = ndimage.binary_dilation(cup, iterations=self.width).astype(cup.dtype)
        eros_cup = ndimage.binary_erosion(cup, iterations=self.width).astype(cup.dtype)
        dila_disc= ndimage.binary_dilation(disc, iterations=self.width).astype(disc.dtype)
        eros_disc= ndimage.binary_erosion(disc, iterations=self.width).astype(disc.dtype)
        cup = dila_cup + eros_cup
        disc = dila_disc + eros_disc
        cup[cup==2]=0
        disc[disc==2]=0
        size = mask.shape
        # boundary = np.zers(size[0:2])
        boundary = (cup + disc) > 0
        return boundary.astype(np.uint8)

class GetBoundary_Single(object):
    def __init__(self, width = 5):
        self.width = width
    def __call__(self, mask):
        dila_mask = ndimage.binary_dilation(mask, iterations=self.width).astype(mask.dtype)
        eros_mask = ndimage.binary_erosion(mask, iterations=self.width).astype(mask.dtype)
        new_mask = dila_mask + eros_mask
        new_mask[new_mask==2]=0
        size = mask.shape
        boundary = new_mask > 0
        return boundary.astype(np.uint8)
    
class GetContourBg(object):
    def __init__(self, bg_width = 5, ct_width = 1):
        self.bg_width = bg_width
        self.ct_width = ct_width
    def __call__(self, mask):
        cup = mask[:, :, 0]
        dila_cup = ndimage.binary_dilation(cup, iterations=self.bg_width).astype(cup.dtype)
        eros_cup = ndimage.binary_erosion(cup, iterations=self.ct_width).astype(cup.dtype)
        cup_contour = cup - eros_cup
        cup_bg = dila_cup - cup

        disc = mask[:, :, 1]
        dila_disc= ndimage.binary_dilation(disc, iterations=self.bg_width).astype(disc.dtype)
        eros_disc= ndimage.binary_erosion(disc, iterations=self.ct_width).astype(disc.dtype)
        disc_contour = disc - eros_disc
        disc_bg = dila_disc - disc

        return cup_contour, cup_bg, disc_contour, disc_bg

class GetContourBg_Single(object):
    def __init__(self, bg_width = 5, ct_width = 1):
        self.bg_width = bg_width
        self.ct_width = ct_width
    def __call__(self, mask):
        dila_mask = ndimage.binary_dilation(mask, iterations=self.bg_width).astype(mask.dtype)
        eros_mask = ndimage.binary_erosion(mask, iterations=self.ct_width).astype(mask.dtype)
        mask_contour = mask - eros_mask
        mask_bg = dila_mask - mask
        return mask_contour, mask_bg


def image_in_painting(image):
    image_np = np.array(image).transpose(2, 0, 1)
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
                                                                      block_noise_size_y) * 255
        cnt -= 1
    image = Image.fromarray(image_np.transpose(1, 2, 0).astype(np.uint8))
    return image

def image_in_painting_constant(image):
    image_np = np.array(image).transpose(2, 0, 1)
    _, image_rows, image_cols = image_np.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(image_rows//6, image_rows//3)
        block_noise_size_y = random.randint(image_cols//6, image_cols//3)
        noise_x = random.randint(3, image_rows-block_noise_size_x-3)
        noise_y = random.randint(3, image_cols-block_noise_size_y-3)
        image_np[:,
                 noise_x:noise_x+block_noise_size_x,
                 noise_y:noise_y+block_noise_size_y] = 255
        cnt -= 1
    image = Image.fromarray(image_np.transpose(1, 2, 0).astype(np.uint8))
    return image

def image_in_painting_rand_constant(image):
    image_np = np.array(image).transpose(2, 0, 1)
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
                                                                block_noise_size_y)) * 255 * random.random()
        cnt -= 1
    image = Image.fromarray(image_np.transpose(1, 2, 0).astype(np.uint8))
    return image


def image_out_painting(image):
    image_np = np.array(image).transpose(2, 0, 1)
    _, image_rows, image_cols = image_np.shape
    image_temp = copy.deepcopy(image_np)
    image_np = np.random.rand(image_np.shape[0], image_np.shape[1], image_np.shape[2]) * 255
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
    image = Image.fromarray(image_np.transpose(1, 2, 0).astype(np.uint8))
    return image

def image_out_painting_constant(image):
    image_np = np.array(image).transpose(2, 0, 1)
    _, image_rows, image_cols = image_np.shape
    image_temp = copy.deepcopy(image_np)
    image_np = np.ones((image_np.shape[0], image_np.shape[1], image_np.shape[2])) * 255
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
    image = Image.fromarray(image_np.transpose(1, 2, 0).astype(np.uint8))
    return image

def image_out_painting_rand_constant(image):
    image_np = np.array(image).transpose(2, 0, 1)
    _, image_rows, image_cols = image_np.shape
    image_temp = copy.deepcopy(image_np)
    image_np = np.ones((image_np.shape[0], image_np.shape[1], image_np.shape[2])) * 255 * random.random()
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
    image = Image.fromarray(image_np.transpose(1, 2, 0).astype(np.uint8))
    return image