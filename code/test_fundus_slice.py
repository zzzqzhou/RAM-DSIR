#!/usr/bin/env python
import argparse
import os
import os.path as osp
import torch.nn as nn

import torch
from tqdm import tqdm
from dataset.fundus import Fundus
from torch.utils.data import DataLoader
import dataset.transform as trans
from torchvision.transforms import Compose
from utils.metrics import *
from dataset import utils
from utils.utils import postprocessing, save_per_img
from test_utils import *
from networks.unet import Encoder, Decoder
import numpy as np
from medpy.metric import binary
from torch.nn import DataParallel
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Test on Fundus dataset (2D slice)')
    # basic settings
    parser.add_argument('--model_file', type=str, default=None, required=True, help='Model path')
    parser.add_argument('--dataset', type=str, default='fundus', help='training dataset')
    parser.add_argument('--data_dir', default='../dataset', help='data root path')
    parser.add_argument('--datasetTest', type=int, default=3, help='test folder id contain images ROIs to test')
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--test_prediction_save_path', type=str, default=None, required=True, help='Path root for test image and mask')
    parser.add_argument('--save_result', action='store_true', help='Save Results')
    parser.add_argument('--freeze_bn', action='store_true', help='Freeze Batch Normalization')
    parser.add_argument('--norm', type=str, default='bn', help='normalization type')
    parser.add_argument('--activation', type=str, default='relu', help='feature activation function')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()
    return args

def main(args):
    data_dir = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(args.test_prediction_save_path):
        os.makedirs(args.test_prediction_save_path)

    model_file = args.model_file
    output_path = os.path.join(args.test_prediction_save_path, 'test' + str(args.datasetTest))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    transform = Compose([trans.Resize((256, 256)), trans.Normalize()])

    testset = Fundus(base_dir=data_dir, split='test', domain_idx=args.datasetTest, transform=transform)
    
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=8,
                            shuffle=False, drop_last=False, pin_memory=True)

    encoder = Encoder(c=args.in_channels, norm=args.norm, activation=args.activation)
    seg_decoder = Decoder(num_classes=args.num_classes, norm=args.norm, activation=args.activation)
    # rec_decoder = Decoder(num_classes=args.in_channels, norm=args.norm, activation=args.activation)
    
    state_dicts = torch.load(model_file)
    
    encoder.load_state_dict(state_dicts['encoder_state_dict'])
    seg_decoder.load_state_dict(state_dicts['seg_decoder_state_dict'])
    
    encoder = DataParallel(encoder).cuda()
    seg_decoder = DataParallel(seg_decoder).cuda()

    if not args.freeze_bn:
        encoder.eval()
        for m in encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
        seg_decoder.eval()
        for m in seg_decoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
    else:
        encoder.eval()
        seg_decoder.eval()

    val_cup_dice = 0.0
    val_disc_dice = 0.0
    total_hd_OC = 0.0
    total_hd_OD = 0.0
    total_asd_OC = 0.0
    total_asd_OD = 0.0
    total_num = 0
    OC = []
    OD = []

    tbar = tqdm(testloader, ncols=150)

    with torch.no_grad():
        for batch_idx, (data, target, target_orgin, ids) in enumerate(tbar):
            data, target = data.cuda(), target.cuda()

            prediction = torch.sigmoid(seg_decoder(encoder(data)))
            prediction = F.interpolate(prediction, size=(target_orgin.size()[2], target_orgin.size()[3]), mode="bilinear")
            data = F.interpolate(data, size=(target_orgin.size()[2], target_orgin.size()[3]), mode="bilinear")

            target_numpy = target_orgin.data.cpu().numpy()
            imgs = data.data.cpu().numpy()

            hd_OC = 100
            asd_OC = 100
            hd_OD = 100
            asd_OD = 100

            for i in range(prediction.shape[0]):
                prediction_post = postprocessing(prediction[i], dataset=args.dataset, threshold=0.75)
                cup_dice, disc_dice = dice_coeff_2label(prediction_post, target_orgin[i])
                OC.append(cup_dice)
                OD.append(disc_dice)
                if np.sum(prediction_post[0, ...]) < 1e-4:
                    hd_OC = 100
                    asd_OC = 100
                else:
                    hd_OC = binary.hd95(np.asarray(prediction_post[0, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
                    asd_OC = binary.asd(np.asarray(prediction_post[0, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
                if np.sum(prediction_post[1, ...]) < 1e-4:
                    hd_OD = 100
                    asd_OD = 100
                else:
                    hd_OD = binary.hd95(np.asarray(prediction_post[1, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 1, ...], dtype=np.bool))

                    asd_OD = binary.asd(np.asarray(prediction_post[1, ...], dtype=np.bool),
                                        np.asarray(target_numpy[i, 1, ...], dtype=np.bool))
                val_cup_dice += cup_dice
                val_disc_dice += disc_dice
                total_hd_OC += hd_OC
                total_hd_OD += hd_OD
                total_asd_OC += asd_OC
                total_asd_OD += asd_OD
                total_num += 1
                if args.save_result:
                    for img, lt, lp in zip([imgs[i]], [target_numpy[i]], [prediction_post]):
                        img, lt = utils.untransform(img, lt)
                        save_per_img(img.transpose(1, 2, 0),
                                    output_path,
                                    ids[i],
                                    lp, lt, mask_path=None, ext="bmp")

        val_cup_dice /= total_num
        val_disc_dice /= total_num
        total_hd_OC /= total_num
        total_asd_OC /= total_num
        total_hd_OD /= total_num
        total_asd_OD /= total_num

        print('''\n==>val_cup_dice : %.2f''' % (100 * val_cup_dice))
        print('''\n==>val_disc_dice : %.2f''' % (100 * val_disc_dice))
        print('''\n==>average_hd_OC : %.2f''' % (total_hd_OC))
        print('''\n==>average_hd_OD : %.2f''' % (total_hd_OD))
        print('''\n==>average_asd_OC : %.2f''' % (total_asd_OC))
        print('''\n==>average_asd_OD : %.2f''' % (total_asd_OD))
        with open(osp.join(output_path, '../test' + str(args.datasetTest) + '_log.csv'), 'a') as f:
            log = [['batch-size: '] + [args.batch_size] + [args.model_file] + \
                   ['cup dice coefficence: '] + [val_cup_dice] + \
                   ['disc dice coefficence: '] + [val_disc_dice] + \
                   ['average_hd_OC: '] + [total_hd_OC] + \
                   ['average_hd_OD: '] + [total_hd_OD] + \
                   ['average_asd_OC: '] + [total_asd_OC] + \
                   ['average_asd_OD: '] + [total_asd_OD]]
            log = map(str, log)
            f.write(','.join(log) + '\n')

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)