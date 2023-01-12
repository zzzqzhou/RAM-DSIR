import os
import argparse
import numpy as np
from networks.unet import Encoder, Decoder, Rec_Decoder
from utils.utils import count_params
from tensorboardX import SummaryWriter
import random
import dataset.transform as trans
from torchvision.transforms import Compose

from dataset.fundus import Fundus_Multi, Fundus
from dataset.prostate import Prostate_Multi
import torch.backends.cudnn as cudnn

from torch.nn import BCELoss, CrossEntropyLoss, DataParallel, KLDivLoss, MSELoss
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from utils.losses import dice_loss, dice_loss_multi
from utils.utils import decode_seg_map_sequence
import shutil
from utils.utils import postprocessing, _connectivity_region_analysis
from utils.metrics import *
import os.path as osp
import SimpleITK as sitk
from medpy.metric import binary
from itertools import cycle

import warnings
warnings.filterwarnings('ignore')

fundus_batch_list = [[3, 6, 7],
                     [2, 7, 7],
                     [2, 4, 10],
                     [2, 4, 10]]

prostate_batch_list = [[2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2]]

def parse_args():
    parser = argparse.ArgumentParser(description='DG Medical Segmentation Train')
    # basic settings
    parser.add_argument('--data_root', type=str, default='../dataset', help='root path of training dataset')
    parser.add_argument('--dataset', type=str, default='fundus', choices=['fundus', 'prostate'], help='training dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of training')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='training epochs')
    parser.add_argument('--domain_idxs', type=str, default='0,1,2', help='training epochs')
    parser.add_argument('--test_domain_idx', type=int, default=3, help='training epochs')
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
    parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    parser.add_argument('--lambda_rec', type=float,  default=0.1, help='lambda of rec')
    parser.add_argument('--deterministic', action='store_true', help='whether use deterministic training')
    parser.add_argument('--ram', action='store_true', help='whether use ram augmentation')
    parser.add_argument('--rec', action='store_true', help='whether use rec loss')
    parser.add_argument('--is_out_domain', action='store_true', help='whether use out domain amp')
    parser.add_argument('--consistency', action='store_true', help='whether use consistency loss')
    parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='path of saved checkpoints')
    parser.add_argument('--norm', type=str, default='bn', help='normalization type')
    parser.add_argument('--activation', type=str, default='relu', help='feature activation function')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()
    return args


domain_list = ['ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC', 'HK']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def KD(input, target):
    consistency_criterion = KLDivLoss()
    loss_consistency = consistency_criterion(input.log(), target) + consistency_criterion(target.log(), input)
    return loss_consistency


def test_fundus(encoder, seg_decoder, epoch, data_dir, datasetTest, output_path, batch_size=8, dataset='fundus'):
    encoder.eval()
    seg_decoder.eval()
    data_dir = os.path.join(data_dir, dataset)
    transform = Compose([trans.Resize((256, 256)), trans.Normalize()])
    testset = Fundus(base_dir=data_dir, split='test',
                     domain_idx=datasetTest, transform=transform)
    
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=8,
                            shuffle=False, drop_last=False, pin_memory=True)
    
    val_cup_dice = 0.0
    val_disc_dice = 0.0
    total_num = 0
    tbar = tqdm(testloader, ncols=150)

    with torch.no_grad():
        for batch_idx, (data, target, target_orgin, ids) in enumerate(tbar):
            data, target = data.cuda(), target.cuda()

            prediction = torch.sigmoid(seg_decoder(encoder(data)))
            prediction = torch.nn.functional.interpolate(prediction, size=(target_orgin.size()[2], target_orgin.size()[3]), mode="bilinear")
            data = torch.nn.functional.interpolate(data, size=(target_orgin.size()[2], target_orgin.size()[3]), mode="bilinear")

            for i in range(prediction.shape[0]):
                prediction_post = postprocessing(prediction[i], dataset=dataset, threshold=0.75)
                cup_dice, disc_dice = dice_coeff_2label(prediction_post, target_orgin[i])
                val_cup_dice += cup_dice
                val_disc_dice += disc_dice
                total_num += 1
        val_cup_dice /= total_num
        val_disc_dice /= total_num

        print('val_cup_dice : {}, val_disc_dice : {}'.format(val_cup_dice, val_disc_dice))
        with open(osp.join(output_path, str(datasetTest) + '_val_log.csv'), 'a') as f:
            log = [['batch-size: '] + [batch_size] + [epoch] + \
                   ['cup dice coefficence: '] + [val_cup_dice] + \
                   ['disc dice coefficence: '] + [val_disc_dice]]
            log = map(str, log)
            f.write(','.join(log) + '\n')
        
        return (val_cup_dice + val_disc_dice) * 100.0 / 2

def test_prostate(encoder, seg_decoder, epoch, data_dir, datasetTest, output_path, batch_size=8, dataset='prostate'):
    encoder.eval()
    seg_decoder.eval()
    domain_name = domain_list[datasetTest]
    data_dir = os.path.join(data_dir, dataset)

    file_list = [item for item in os.listdir(os.path.join(data_dir, domain_name)) if 'segmentation' not in item]

    tbar = tqdm(file_list, ncols=150)

    val_dice = 0.0
    total_num = 0
    for file_name in tbar:
        itk_image = sitk.ReadImage(os.path.join(data_dir, domain_name, file_name))
        itk_mask = sitk.ReadImage(os.path.join(data_dir, domain_name, file_name.replace('.nii.gz', '_segmentation.nii.gz')))

        image = sitk.GetArrayFromImage(itk_image)
        mask = sitk.GetArrayFromImage(itk_mask)

        max_value = np.max(image)
        min_value = np.min(image)
        image = 2 * (image - min_value) / (max_value - min_value) - 1

        mask[mask==2] = 1
        pred_y = np.zeros(mask.shape)

        #### channel 3 ####
        frame_list = [kk for kk in range(1, image.shape[0] - 1)]

        for ii in range(int(np.floor(image.shape[0] // batch_size))):
            vol = np.zeros([batch_size, 3, image.shape[1], image.shape[2]])

            for idx, jj in enumerate(frame_list[ii * batch_size : (ii + 1) * batch_size]):
                vol[idx, ...] = image[jj - 1 : jj + 2, ...].copy()
            vol = torch.from_numpy(vol).float().cuda()

            pred_student = torch.max(torch.softmax(seg_decoder(encoder(vol)), dim=1), dim=1)[1].detach().data.cpu().numpy()

            for idx, jj in enumerate(frame_list[ii * batch_size : (ii + 1) * batch_size]):
                ###### Ignore slices without prostate region ######
                if np.sum(mask[jj, ...]) == 0:
                    continue
                pred_y[jj, ...] = pred_student[idx, ...].copy()

        
        processed_pred_y = _connectivity_region_analysis(pred_y)
        dice_coeff = binary.dc(np.asarray(processed_pred_y, dtype=np.bool),
                            np.asarray(mask, dtype=np.bool))
        val_dice += dice_coeff
        total_num += 1
    
    val_dice /= total_num
    print('val_dice : {}'.format(val_dice))
    with open(osp.join(output_path, str(datasetTest) + '_val_log.csv'), 'a') as f:
            log = [['batch-size: '] + [batch_size] + [epoch] + \
                   ['dice coefficence: '] + [val_dice]]
            log = map(str, log)
            f.write(','.join(log) + '\n')
    return val_dice * 100.0


def train_fundus(trainloader_list, encoder, seg_decoder, rec_decoder, writer, args, optimizer, dataloader_length_max, domain_idx_list, batch_size_list):
    if args.consistency_type == 'mse':
        consistency_criterion = MSELoss()
    elif args.consistency_type == 'kd':
        consistency_criterion = KD
    else:
        assert False, args.consistency_type
    criterion = BCELoss()
    rec_criterion = MSELoss()

    encoder = DataParallel(encoder).cuda()
    seg_decoder = DataParallel(seg_decoder).cuda()
    if args.rec:
        rec_decoder = DataParallel(rec_decoder).cuda()

    total_iters = dataloader_length_max * args.epochs

    previous_best = 0.0
    iter_num = 0
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.6f" %
              (epoch, optimizer.param_groups[0]["lr"]))
        
        encoder.train()
        seg_decoder.train()
        if args.rec:
            rec_decoder.train()

        tbar = tqdm(zip(*trainloader_list), ncols=150)

        for i, sample_batches in enumerate(tbar):
            img_multi = None
            img_freq_multi = None
            mask_multi = None
            rec_soft_multi = None
            avg_rec_loss = 0.0

            for train_idx in range(len(domain_idx_list)):
                img, img_freq, mask = sample_batches[train_idx][0], sample_batches[train_idx][1], sample_batches[train_idx][2]

                if img_multi is None:
                    img_multi = img
                    img_freq_multi = img_freq
                    mask_multi = mask
                else:
                    img_multi = torch.cat([img_multi, img], 0)
                    img_freq_multi = torch.cat([img_freq_multi, img_freq], 0)
                    mask_multi = torch.cat([mask_multi, mask], 0)
            
            img_multi, img_freq_multi, mask_multi = img_multi.cuda(), img_freq_multi.cuda(), mask_multi.cuda()

            img_feats = encoder(img_multi)
            pred_soft_1 = torch.sigmoid(seg_decoder(img_feats))
            loss_bce_1 = criterion(pred_soft_1, mask_multi)
            loss_dice_1 = dice_loss(pred_soft_1, mask_multi)
            
            loss = 0
            if args.ram:
                img_freq_feats = encoder(img_freq_multi)
                pred_soft_2 = torch.sigmoid(seg_decoder(img_freq_feats))
                loss_bce_2 = criterion(pred_soft_2, mask_multi)
                loss_dice_2 = dice_loss(pred_soft_2, mask_multi)

                if args.consistency:
                    loss_consistency = consistency_criterion(pred_soft_2, pred_soft_1)
                else:
                    loss_consistency = 0
                
                if args.rec:
                    left = 0
                    for train_idx in range(len(domain_idx_list)):
                        right = left + batch_size_list[train_idx]
                        rec_soft = torch.tanh(rec_decoder(img_freq_feats[-1][left:right, ...], 
                                            domain_label=train_idx*torch.ones(batch_size_list[train_idx], dtype=torch.long)))
                        if rec_soft_multi is None:
                            rec_soft_multi = rec_soft
                        else:
                            rec_soft_multi = torch.cat([rec_soft_multi, rec_soft], 0)
                        loss_rec = rec_criterion(rec_soft, img_multi[left:right])
                        loss = loss + args.lambda_rec * loss_rec
                        avg_rec_loss += loss_rec.item()
                        left = right
            
            else:
                loss_bce_2 = 0
                loss_dice_2 = 0
                loss_consistency = 0

            loss = loss + loss_bce_1 + loss_bce_2 + loss_dice_1 + loss_dice_2 + 0.5 * loss_consistency

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = args.lr * (1 - iter_num / total_iters) ** 0.9
            if args.rec:
                optimizer.param_groups[0]["lr"] = lr / 2
                optimizer.param_groups[1]["lr"] = lr
                optimizer.param_groups[2]["lr"] = lr
            else:
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr

            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('loss/loss_bce_1', loss_bce_1, iter_num)
            writer.add_scalar('loss/loss_dice_1', loss_dice_1, iter_num)
            writer.add_scalar('loss/loss_bce_2', loss_bce_2, iter_num)
            writer.add_scalar('loss/loss_dice_2', loss_dice_2, iter_num)
            writer.add_scalar('loss/loss_consistency', loss_consistency, iter_num)
            writer.add_scalar('loss/loss_rec', avg_rec_loss / 4, iter_num)

            if iter_num  % 100 == 0:
                image = img_multi[0:9:4, 0:3, ...]
                grid_image = make_grid(image, 3, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = img_freq_multi[0:9:4, 0:3, ...]
                grid_image = make_grid(image, 3, normalize=True)
                writer.add_image('train/Image_Freq', grid_image, iter_num)

                image = rec_soft_multi[0:9:4, 0:3, ...]
                grid_image = make_grid(image, 3, normalize=True)
                writer.add_image('train/Image_Rec', grid_image, iter_num)

                grid_image = make_grid(pred_soft_1[0:9:4, 0, ...].unsqueeze(1), 3, normalize=True)
                writer.add_image('train/Soft_Predicted_OC', grid_image, iter_num)

                grid_image = make_grid(pred_soft_1[0:9:4, 1, ...].unsqueeze(1), 3, normalize=True)
                writer.add_image('train/Soft_Predicted_OD', grid_image, iter_num)

                grid_image = make_grid(mask_multi[0:9:4, 0, ...].unsqueeze(1), 3, normalize=False)
                writer.add_image('train/GT_OC', grid_image, iter_num)

                grid_image = make_grid(mask_multi[0:9:4, 1, ...].unsqueeze(1), 3, normalize=False)
                writer.add_image('train/GT_OD', grid_image, iter_num)
            
            iter_num = iter_num + 1

        if (epoch + 1) % 1 == 0:
            print("Test on target domain {}".format(args.test_domain_idx))
            with torch.no_grad():
                avg_dice = test_fundus(encoder, seg_decoder, epoch, args.data_root, args.test_domain_idx, args.save_path, args.test_batch_size, dataset=args.dataset)
            if avg_dice >= previous_best:
                if previous_best != 0:
                    model_path = os.path.join(args.save_path, 'model_%.2f.pth' % (previous_best))
                    if os.path.exists(model_path):
                        os.remove(model_path)
                if args.rec:
                    checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                                  "seg_decoder_state_dict": seg_decoder.module.state_dict(),
                                  "rec_decoder_state_dict": rec_decoder.module.state_dict()}
                else:
                    checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                                  "seg_decoder_state_dict": seg_decoder.module.state_dict()}
                torch.save(checkpoint, os.path.join(args.save_path, 'model_%.2f.pth' % (avg_dice)))
                previous_best = avg_dice
                
    save_mode_path = os.path.join(args.save_path, 'final_model.pth')
    if args.rec:
        checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                      "seg_decoder_state_dict": seg_decoder.module.state_dict(),
                      "rec_decoder_state_dict": rec_decoder.module.state_dict()}
    else:
        checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                      "seg_decoder_state_dict": seg_decoder.module.state_dict()}
    torch.save(checkpoint, save_mode_path)
    print('\nSave Final Model to {}'.format(args.save_path))

def train_prostate(trainloader_list, encoder, seg_decoder, rec_decoder, writer, args, optimizer, dataloader_length_max, domain_idx_list, batch_size_list):
    if args.consistency_type == 'mse':
        consistency_criterion = MSELoss()
    elif args.consistency_type == 'kd':
        consistency_criterion = KD
    else:
        assert False, args.consistency_type
    criterion = CrossEntropyLoss()
    rec_criterion = MSELoss()

    encoder = DataParallel(encoder).cuda()
    seg_decoder = DataParallel(seg_decoder).cuda()
    if args.rec:
        rec_decoder = DataParallel(rec_decoder).cuda()

    total_iters = dataloader_length_max * args.epochs

    previous_best = 0.0
    iter_num = 0
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.6f" %
              (epoch, optimizer.param_groups[0]["lr"]))
        
        encoder.train()
        seg_decoder.train()
        if args.rec:
            rec_decoder.train()

        tbar = tqdm(zip(*trainloader_list), ncols=150)

        for i, sample_batches in enumerate(tbar):
            img_multi = None
            img_freq_multi = None
            mask_multi = None
            rec_soft_multi = None
            avg_rec_loss = 0.0
            for train_idx in range(len(domain_idx_list)):
                img, img_freq, mask = sample_batches[train_idx][0], sample_batches[train_idx][1], sample_batches[train_idx][2]

                if img_multi is None:
                    img_multi = img
                    img_freq_multi = img_freq
                    mask_multi = mask
                else:
                    img_multi = torch.cat([img_multi, img], 0)
                    img_freq_multi = torch.cat([img_freq_multi, img_freq], 0)
                    mask_multi = torch.cat([mask_multi, mask], 0)
            
            img_multi, img_freq_multi, mask_multi = img_multi.cuda(), img_freq_multi.cuda(), mask_multi.cuda()

            img_feats = encoder(img_multi)
            pred_1 = seg_decoder(img_feats)
            pred_soft_1 = torch.softmax(pred_1, dim=1)
            loss_ce_1 = criterion(pred_1, mask_multi)
            loss_dice_1 = dice_loss_multi(pred_soft_1, mask_multi, num_classes=args.num_classes, ignore_index=0)

            loss = 0
            if args.ram:
                img_freq_feats = encoder(img_freq_multi)
                pred_2 = seg_decoder(img_freq_feats)
                pred_soft_2 = torch.softmax(pred_2, dim=1)
                loss_ce_2 = criterion(pred_2, mask_multi)
                loss_dice_2 = dice_loss_multi(pred_soft_2, mask_multi, num_classes=args.num_classes, ignore_index=0)
                
                if args.consistency:
                    loss_consistency = consistency_criterion(pred_soft_2, pred_soft_1)
                else:
                    loss_consistency = 0
                
                if args.rec:
                    left = 0
                    for train_idx in range(len(domain_idx_list)):
                        right = left + batch_size_list[train_idx]
                        rec_soft = torch.tanh(rec_decoder(img_freq_feats[-1][left:right, ...], 
                                                        domain_label=train_idx*torch.ones(batch_size_list[train_idx], dtype=torch.long)))
                        if rec_soft_multi is None:
                            rec_soft_multi = rec_soft
                        else:
                            rec_soft_multi = torch.cat([rec_soft_multi, rec_soft], 0)
                        loss_rec = rec_criterion(rec_soft, img_multi[left:right, ...])
                        loss = loss + args.lambda_rec * loss_rec
                        avg_rec_loss += loss_rec.item()
                        left = right
            
            else:
                loss_ce_2 = 0
                loss_dice_2 = 0
                loss_consistency = 0

            loss = loss + loss_ce_1 + loss_ce_2 + loss_dice_1 + loss_dice_2 + 0.5 * loss_consistency
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = args.lr * (1 - iter_num / total_iters) ** 0.9
            if args.rec:
                optimizer.param_groups[0]["lr"] = lr / 2
                optimizer.param_groups[1]["lr"] = lr
                optimizer.param_groups[2]["lr"] = lr
            else:
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr

            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('loss/loss_ce_1', loss_ce_1, iter_num)
            writer.add_scalar('loss/loss_dice_1', loss_dice_1, iter_num)
            writer.add_scalar('loss/loss_ce_2', loss_ce_2, iter_num)
            writer.add_scalar('loss/loss_dice_2', loss_dice_2, iter_num)
            writer.add_scalar('loss/loss_consistency', loss_consistency, iter_num)
            writer.add_scalar('loss/loss_rec', avg_rec_loss / 4, iter_num)

            if iter_num  % 100 == 0:
                image = img_multi[0:7:3, 1, ...].unsqueeze(1) # channel 3
                grid_image = make_grid(image, 3, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = img_freq_multi[0:7:3, 1, ...].unsqueeze(1) # channel 3
                grid_image = make_grid(image, 3, normalize=True)
                writer.add_image('train/Image_Freq', grid_image, iter_num)

                image = rec_soft_multi[0:7:3, 1, ...].unsqueeze(1) # channel 3
                grid_image = make_grid(image, 3, normalize=True)
                writer.add_image('train/Image_Rec', grid_image, iter_num)

                image = torch.max(pred_soft_1[0:7:3, ...], 1)[1].detach().data.cpu().numpy()
                image = decode_seg_map_sequence(image)
                grid_image = make_grid(image, 3, normalize=False)
                writer.add_image('train/Predicted', grid_image, iter_num)

                image = mask_multi[0:7:3, ...].detach().data.cpu().numpy()
                image = decode_seg_map_sequence(image)
                grid_image = make_grid(image, 3, normalize=False)
                writer.add_image('train/GT', grid_image, iter_num)
            
            iter_num = iter_num + 1
        
        if (epoch + 1) % 1 == 0:
            print("Test on target domain {}".format(args.test_domain_idx))
            with torch.no_grad():
                avg_dice = test_prostate(encoder, seg_decoder, epoch, args.data_root, args.test_domain_idx, args.save_path, args.test_batch_size, dataset=args.dataset)
            if avg_dice >= previous_best:
                if previous_best != 0:
                    model_path = os.path.join(args.save_path, 'model_%.2f.pth' % (previous_best))
                    if os.path.exists(model_path):
                        os.remove(model_path)
                if args.rec:
                    checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                                "seg_decoder_state_dict": seg_decoder.module.state_dict(),
                                "rec_decoder_state_dict": rec_decoder.module.state_dict()}
                else:
                    checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                                "seg_decoder_state_dict": seg_decoder.module.state_dict()}
                torch.save(checkpoint, os.path.join(args.save_path, 'model_%.2f.pth' % (avg_dice)))
                previous_best = avg_dice
                
    save_mode_path = os.path.join(args.save_path, 'final_model.pth')
    if args.rec:
        checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                    "seg_decoder_state_dict": seg_decoder.module.state_dict(),
                    "rec_decoder_state_dict": rec_decoder.module.state_dict()}
    else:
        checkpoint = {"encoder_state_dict": encoder.module.state_dict(),
                    "seg_decoder_state_dict": seg_decoder.module.state_dict()}
    torch.save(checkpoint, save_mode_path)
    print('\nSave Final Model to {}'.format(args.save_path))

def main(args):
    data_root = os.path.join(args.data_root, args.dataset)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if os.path.exists(args.save_path + '/code'):
        shutil.rmtree(args.save_path + '/code')
    shutil.copytree('.', args.save_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))
    
    writer = SummaryWriter(args.save_path + '/log')

    dataset_zoo = {'fundus': Fundus_Multi, 'prostate': Prostate_Multi}
    transform = {'fundus': Compose([trans.Resize((256, 256)), trans.RandomScaleCrop((256, 256))]),
                 'prostate': None}
    batch_size_list = {'fundus': fundus_batch_list[args.test_domain_idx] if args.test_domain_idx < 4 else None,
                       'prostate': prostate_batch_list[args.test_domain_idx]}

    domain_idx_list = args.domain_idxs.split(',')
    domain_idx_list = [int(item) for item in domain_idx_list]

    dataloader_list = []

    dataloader_length_max = -1
    max_id = 0
    max_dataloader = None
    count = 0
    for idx, i in enumerate(domain_idx_list):
        trainset = dataset_zoo[args.dataset](base_dir=data_root, split='train',
                            domain_idx_list=[i], transform=transform[args.dataset], is_out_domain=args.is_out_domain, test_domain_idx=args.test_domain_idx)
        trainloader = DataLoader(trainset, batch_size=batch_size_list[args.dataset][idx], num_workers=8,
                             shuffle=True, drop_last=True, pin_memory=True, worker_init_fn=seed_worker)
        dataloader_list.append(cycle(trainloader))
        if dataloader_length_max < len(trainloader):
            dataloader_length_max = len(trainloader)
            max_dataloader = trainloader
            max_id = count
        count += 1
    dataloader_list[max_id] = max_dataloader
    
    encoder = Encoder(c=args.in_channels, norm=args.norm, activation=args.activation)
    seg_decoder = Decoder(num_classes=args.num_classes, norm=args.norm, activation=args.activation)
    if args.rec:
        if args.dataset == 'fundus':
            rec_decoder = Rec_Decoder(num_classes=args.in_channels, norm='dsbn', activation=args.activation, num_domains=len(domain_idx_list))
            optimizer = Adam([{"params": encoder.parameters(), 'lr': args.lr / 2},
                            {"params": seg_decoder.parameters(), 'lr': args.lr},
                            {"params": rec_decoder.parameters(), 'lr': args.lr}],
                            lr=args.lr, betas=(0.9, 0.999))
        else:
            rec_decoder = Rec_Decoder(num_classes=args.in_channels, norm='dsbn', activation=args.activation, num_domains=len(domain_idx_list))
            optimizer = Adam([{"params": encoder.parameters(), 'lr': args.lr / 2},
                            {"params": seg_decoder.parameters(), 'lr': args.lr},
                            {"params": rec_decoder.parameters(), 'lr': args.lr}],
                            lr=args.lr, betas=(0.9, 0.999))
    else:
        rec_decoder = None
        optimizer = Adam([{"params": encoder.parameters(), 'lr': args.lr},
                          {"params": seg_decoder.parameters(), 'lr': args.lr}],
                          lr=args.lr, betas=(0.9, 0.999))

    print('\nEncoder Params: %.3fM' % count_params(encoder))
    print('\nSeg Decoder Params: %.3fM' % count_params(seg_decoder))
    print('\nRec Decoder Params: %.3fM' % count_params(rec_decoder))
    

    if args.dataset == 'fundus':
        train_fundus(dataloader_list, encoder, seg_decoder, rec_decoder, writer,
                     args, optimizer, dataloader_length_max, domain_idx_list, batch_size_list[args.dataset])
    elif args.dataset == 'prostate':
        train_prostate(dataloader_list, encoder, seg_decoder, rec_decoder, writer,
                       args, optimizer, dataloader_length_max, domain_idx_list, batch_size_list[args.dataset])
    else:
        raise ValueError('Not support Dataset {}'.format(args.dataset))


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.epochs is None:
        args.epochs = {'fundus': 400, 'prostate': 200}[args.dataset]
    if args.lr is None:
        args.lr = {'fundus': 2e-3, 'prostate': 1e-3}[args.dataset]
    if args.num_classes is None:
        args.num_classes = {'fundus': 2, 'prostate': 2}[args.dataset]

    print(args)

    main(args)