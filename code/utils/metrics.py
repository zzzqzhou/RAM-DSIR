import numpy as np
from medpy import metric


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def dice(input, target, ignore_index=None):
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()
    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

def dice_multi(input, target, num_classes=3, ignore_index=None):
    smooth = 1e-5
    count = 0
    total_dice = 0
    for i in range(num_classes):
        if i == ignore_index:
            continue
        count += 1
        intersect = ((input == i) * (target == i)).sum()
        y_sum = ((target == i) * (target == i)).sum()
        z_sum = ((input == i) * (input == i)).sum()
        dice_i = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        total_dice += dice_i
    return total_dice / count

def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value

def dice_coeff_2label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    # pred = torch.sigmoid(pred)
    # pred = pred.data.cpu()
    # pred[pred > 0.75] = 1
    # pred[pred <= 0.75] = 0
    # print target.shape
    # print pred.shape
    if len(pred.shape) == 3:
        return dice_coefficient_numpy(pred[0, ...], target[0, ...]), dice_coefficient_numpy(pred[1, ...], target[1, ...])
    else:
        dice_cup = []
        dice_disc = []
        for i in range(pred.shape[0]):
            cup, disc = dice_coefficient_numpy(pred[i, 0, ...], target[i, 0, ...]), dice_coefficient_numpy(pred[i, 1, ...], target[i, 1, ...])
            dice_cup.append(cup)
            dice_disc.append(disc)
    return sum(dice_cup) / len(dice_cup), sum(dice_disc) / len(dice_disc)

def dice_coeff_1label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()

    if len(pred.shape) == 2:
        return dice_coefficient_numpy(pred, target)
    elif len(pred.shape) == 3:
        return dice_coefficient_numpy(pred[0, ...], target[0, ...])
    else:
        dice_list = []
        for i in range(pred.shape[0]):
            dice_coeff = dice_coefficient_numpy(pred[i, 0, ...], target[i, 0, ...])
            dice_list.append(dice_coeff)
    
    return sum(dice_list) / len(dice_list)