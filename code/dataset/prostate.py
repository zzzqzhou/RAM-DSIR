import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)
    
    fft = np.fft.fft2( img_np, axes=(-2, -1) )
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

    _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)
    # print (b)
    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    ratio = random.randint(1,10)/10

    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    # a_src[:,h1:h2,w1:w2] = a_trg[:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    # a_trg[:,h1:h2,w1:w2] = a_src[:,h1:h2,w1:w2]
    # a_trg = np.fft.ifftshift( a_trg, axes=(-2, -1) )
    return a_src

def source_to_target_freq( src_img, amp_trg, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img
    src_img = src_img.transpose((2, 0, 1))
    src_img_np = src_img #.cpu().numpy()
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg.transpose(1, 2, 0)

class Prostate(Dataset):
    def __init__(self, domain_idx=None, base_dir=None, split='train', num=None, transform=None):
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5', 'Domain6']
        self.domain_idx = domain_idx
        self.split = split

        self.id_path = os.listdir(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'image'))

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))
    
    def __len__(self):
        return len(self.id_path)
    
    def __getitem__(self, index):
        id = self.id_path[index]
        img = np.load(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'image', id))

        if self.split == 'test':
            mask = np.load(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'mask', id))
            sample = {'img': img, 'mask': mask}
            
            img = sample['img']
            mask = sample['mask']
            img = img.transpose(2, 0, 1)

            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()
            
            if 'onehot_label' in sample.keys():
                onehot_label = sample['onehot_label']
                onehot_label = torch.from_numpy(onehot_label).long()
                return img, mask, onehot_label, id.split('/')[-1]

            return img, mask, id.split('/')[-1]
        
        else:
            mask = np.load(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'mask', id))
            sample = {'img': img, 'mask': mask}
            
            img = sample['img']
            mask = sample['mask']
            img = img.transpose(2, 0, 1)
            
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask)#.long()
            
            if 'onehot_label' in sample.keys():
                onehot_label = sample['onehot_label']
                onehot_label = torch.from_numpy(onehot_label).long()
                return img, mask.long(), onehot_label
            return img, mask.long()


class Prostate_Multi(Dataset):
    def __init__(self, domain_idx_list=None, base_dir=None, split='train', num=None, transform=None, is_freq=True, is_out_domain=False, test_domain_idx=None):
        self.base_dir = base_dir
        self.num = num
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5', 'Domain6']
        self.domain_idx_list = domain_idx_list
        self.split = split
        self.is_freq = is_freq
        self.is_out_domain = is_out_domain
        self.test_domain_idx = test_domain_idx

        self.id_path = []
        for domain_idx in self.domain_idx_list:
            domain_list = os.listdir(os.path.join(self.base_dir, self.domain_name[domain_idx], 'image'))
            domain_list = [self.domain_name[domain_idx] + '/image/' + item for item in domain_list]
            self.id_path = self.id_path + domain_list

        if self.num is not None:
            self.id_path = self.id_path[:self.num]
        print("total {} samples".format(len(self.id_path)))
    
    def __len__(self):
        return len(self.id_path)
    
    def __getitem__(self, index):
        train_domain_name = self.domain_name.copy()
        train_domain_name.remove(self.domain_name[self.test_domain_idx])
        id = self.id_path[index]
        img = np.load(os.path.join(self.base_dir, id))
        cur_domain_name = id.split('/')[0]

        if self.split == 'test':
            mask = np.load(os.path.join(self.base_dir, id.replace('image', 'mask')))
            sample = {'img': img, 'mask': mask}
            
            img = sample['img']
            mask = sample['mask']
            img = img.transpose(2, 0, 1)

            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).long()
            
            if 'onehot_label' in sample.keys():
                onehot_label = sample['onehot_label']
                onehot_label = torch.from_numpy(onehot_label).long()
                return img, mask, onehot_label, id.split('/')[-1]

            return img, mask, id.split('/')[-1]
        
        else:
            mask = np.load(os.path.join(self.base_dir, id.replace('image', 'mask')))
            sample = {'img': img, 'mask': mask}
            
            img = sample['img']
            mask = sample['mask']

            if self.is_freq:
                domain_list = train_domain_name.copy()
                if self.is_out_domain:
                    domain_list.remove(cur_domain_name)
                # print(domain_list)
                other_domain_name = np.random.choice(domain_list, 1)[0]
                other_id =np.random.choice(os.listdir(os.path.join(self.base_dir, other_domain_name, 'image')))
                other_img =  np.load(os.path.join(self.base_dir, other_domain_name, 'image', other_id))

                amp_trg = extract_amp_spectrum(other_img.transpose(2, 0, 1))
                img_freq = source_to_target_freq(img, amp_trg, L=0.1)
                img_freq = np.clip(img_freq, -1, 1)

                img_freq = img_freq.transpose(2, 0, 1)

                img = img.transpose(2, 0, 1)
                
                img = torch.from_numpy(img).float()
                img_freq = torch.from_numpy(img_freq).float()
                mask = torch.from_numpy(mask)#.long()
                
                if 'onehot_label' in sample.keys():
                    onehot_label = sample['onehot_label']
                    onehot_label = torch.from_numpy(onehot_label).long()
                    return img, img_freq, mask.long(), onehot_label
                return img, img_freq, mask.long()
            
            else:
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).float()
                mask = torch.from_numpy(mask)#.long()
                if 'onehot_label' in sample.keys():
                    onehot_label = sample['onehot_label']
                    onehot_label = torch.from_numpy(onehot_label).long()
                    return img, mask.long(), onehot_label
                return img, mask.long()


# if __name__ == '__main__':
#     import transform as trans
#     from torch.utils.data.dataloader import DataLoader

#     base_dir = '/data/ziqi/datasets/muti_site_med/prostate'
#     trainset = Prostate_Multi(base_dir=base_dir,
#                           split='train',
#                           domain_idx_list=[0],
#                           is_out_domain=True,
#                           test_domain_idx=4)
    
#     trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
#     for i, (img, img_freq, mask, mask_2, mask_4, mask_8, mask_16) in enumerate(trainloader):
#         print(img.shape, img_freq.shape)