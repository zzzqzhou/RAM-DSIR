import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os.path as osp
import os
import cv2
from skimage import morphology
import scipy
from matplotlib.pyplot import imsave
# from keras.preprocessing import image
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from skimage import measure, draw
from skimage.morphology import disk, erosion, dilation, opening, closing, white_tophat
from scipy import ndimage
plt.switch_backend('agg')

def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def _connectivity_region_analysis(mask):
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    label_im, nb_labels = ndimage.label(mask)#, structure=s)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)        
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im


def postprocessing(prediction, threshold=0.50, dataset='G'):
    if dataset == 'prostate':
        prediction = prediction.data.cpu().numpy()
        prediction_copy = np.copy(prediction)
        prediction_copy = prediction_copy.astype(np.uint8)
        mask = get_largest_fillhole(prediction_copy).astype(np.uint8)  # return 0,1
        # mask = _connectivity_region_analysis(mask)
        return mask
    elif dataset[0] == 'D':
        # prediction = prediction.numpy()
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        disc_mask = (disc_mask > 0.5)  # return binary mask
        cup_mask = (cup_mask > 0.1)  # return binary mask
        disc_mask = disc_mask.astype(np.uint8)
        cup_mask = cup_mask.astype(np.uint8)
        # for i in range(5):
        #     disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        #     cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        # cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        return prediction_copy
    else:
        # prediction = torch.sigmoid(prediction).data.cpu().numpy()
        prediction = prediction.data.cpu().numpy()

        # disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        # cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.erosion(disc_mask, morphology.diamond(3))  # return 0,1
        # cup_mask = morphology.erosion(cup_mask, morphology.diamond(3))  # return 0,1

        prediction_copy = np.copy(prediction)
        prediction_copy = (prediction_copy > threshold)  # return binary mask
        prediction_copy = prediction_copy.astype(np.uint8)
        disc_mask = prediction_copy[1]
        cup_mask = prediction_copy[0]
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        # selem = disk(6)
        # disc_mask = morphology.closing(disc_mask, selem)
        # cup_mask = morphology.closing(cup_mask, selem)
        # print(sum(disc_mask))


        return prediction_copy


def joint_val_image(image, prediction, mask):
    ratio = 0.5
    _pred_cup = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _pred_disc = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _mask = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    image = np.transpose(image, (1, 2, 0))

    _pred_cup[:, :, 0] = prediction[0]
    _pred_cup[:, :, 1] = prediction[0]
    _pred_cup[:, :, 2] = prediction[0]
    _pred_disc[:, :, 0] = prediction[1]
    _pred_disc[:, :, 1] = prediction[1]
    _pred_disc[:, :, 2] = prediction[1]
    _mask[:,:,0] = mask[0]
    _mask[:,:,1] = mask[1]

    pred_cup = np.add(ratio * image, (1 - ratio) * _pred_cup)
    pred_disc = np.add(ratio * image, (1 - ratio) * _pred_disc)
    mask_img = np.add(ratio * image, (1 - ratio) * _mask)

    joint_img = np.concatenate([image, mask_img, pred_cup, pred_disc], axis=1)
    return joint_img


def save_val_img(path, epoch, img):
    name = osp.join(path, "visualization", "epoch_%d.png" % epoch)
    out = osp.join(path, "visualization")
    if not osp.exists(out):
        os.makedirs(out)
    img_shape = img[0].shape
    stack_image = np.zeros([len(img) * img_shape[0], img_shape[1], img_shape[2]])
    for i in range(len(img)):
        stack_image[i * img_shape[0] : (i + 1) * img_shape[0], :, : ] = img[i]
    imsave(name, stack_image)





def save_per_img(patch_image, data_save_path, img_name, prob_map, gt=None, mask_path=None, ext="bmp"):
    # path1 = os.path.join(data_save_path, 'overlay', img_name.split('.')[0]+'.png')
    path1 = os.path.join(data_save_path, img_name.split('.')[0].split('/')[-1]+'.png')
    # path0 = os.path.join(data_save_path, 'original_image', img_name.split('.')[0]+'.png')
    # if not os.path.exists(os.path.dirname(path0)):
    #     os.makedirs(os.path.dirname(path0))
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    disc_map = prob_map[0]
    cup_map = prob_map[1]
    size = disc_map.shape
    disc_map[:, 0] = np.zeros(size[0])
    disc_map[:, size[1] - 1] = np.zeros(size[0])
    disc_map[0, :] = np.zeros(size[1])
    disc_map[size[0] - 1, :] = np.zeros(size[1])
    size = cup_map.shape
    cup_map[:, 0] = np.zeros(size[0])
    cup_map[:, size[1] - 1] = np.zeros(size[0])
    cup_map[0, :] = np.zeros(size[1])
    cup_map[size[0] - 1, :] = np.zeros(size[1])


    contours_disc = measure.find_contours(disc_map, 0.5)
    contours_cup = measure.find_contours(cup_map, 0.5)


    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]

    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]

    disc_mask = get_largest_fillhole(gt[0]).astype(np.uint8)  # return 0,1
    cup_mask = get_largest_fillhole(gt[1]).astype(np.uint8)

    contours_disc = measure.find_contours(disc_mask, 0.5)
    contours_cup = measure.find_contours(cup_mask, 0.5)
    red = [255, 0, 0]
    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red

    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red


    patch_image = patch_image.astype(np.uint8)
    patch_image = Image.fromarray(patch_image)

    patch_image.save(path1)


def save_per_img_prostate(patch_image, data_save_path, img_name, prob_map, gt=None, mask_path=None, ext="bmp"):
    # path1 = os.path.join(data_save_path, 'overlay', img_name.split('.')[0]+'.png')
    path1 = os.path.join(data_save_path, img_name.split('.')[0].split('/')[-1]+'.png')
    # path0 = os.path.join(data_save_path, 'original_image', img_name.split('.')[0]+'.png')
    # if not os.path.exists(os.path.dirname(path0)):
    #     os.makedirs(os.path.dirname(path0))
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    # map = prob_map[0]
    map = prob_map
    size = map.shape
    map[:, 0] = np.zeros(size[0])
    map[:, size[1] - 1] = np.zeros(size[0])
    map[0, :] = np.zeros(size[1])
    map[size[0] - 1, :] = np.zeros(size[1])

    contours = measure.find_contours(map, 0.5)

    for n, contour in enumerate(contours):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]

    # mask = get_largest_fillhole(gt[0]).astype(np.uint8)  # return 0,1
    mask = get_largest_fillhole(gt).astype(np.uint8)  # return 0,1
    
    contours = measure.find_contours(mask, 0.5)
    red = [255, 0, 0]
    for n, contour in enumerate(contours):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red


    patch_image = patch_image.astype(np.uint8)
    patch_image = Image.fromarray(patch_image)

    patch_image.save(path1)

def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6