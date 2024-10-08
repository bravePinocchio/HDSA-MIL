import os
gpu_index = (1,)
gpu_ids = tuple(gpu_index)
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
from models.my_model import My_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms
import cv2 as cv

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings

class BagDataset():
    def __init__(self, csv_file, pretrained = True):
        self.files_list = csv_file
        self.transform = self.get_trans(pretrained)

    def get_trans(self, pretrained):
        if pretrained:
            mean = (0.7219808, 0.58108766, 0.70086701)
            std = (0.18555619, 0.2276182, 0.18211643)
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        trans = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return trans

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        img = self.transform(img)
        img_name = path.split(os.sep)[-1]
        img_pos = np.asarray(
            [int(img_name.split('.')[0].split('_')[0]), int(img_name.split('.')[0].split('_')[1])])  # row, col
        sample = {'input': img, 'position': img_pos}

        return sample



def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def test(args, bags_list, model, resnet):
    model.eval()
    resnet.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    #colors = [np.random.choice(range(256), size=3) for i in range(args.n_classes)]
    colors_1 = np.array([0,256,256])
    for i in range(0, num_bags):
        feats_list = []
        pos_list = []
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.' + args.patch_ext))
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']
                feats = get_feature(resnet, patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                pos_list.extend(patch_pos)
            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            bag_feats = torch.from_numpy(feats_arr).cuda()

            A = model(bag_feats, attention_only = True)
            A = torch.transpose(A, 1, 0)
            A = F.softmax(A, dim=0)

            for c in range(args.n_classes):
                attentions = A[:, c].cpu().numpy()
                #colored_tiles = np.matmul(attentions[:, None], colors[0][None, :])
                colored_tiles = np.matmul(attentions[:, None], colors_1[None, :])
                colored_tiles = exposure.rescale_intensity(colored_tiles, out_range=(0, 1))
                color_map = np.zeros((np.amax(pos_arr, 0)[0] + 1, np.amax(pos_arr, 0)[1] + 1, 3))
                for k, pos in enumerate(pos_arr):
                    color_map[pos[0], pos[1]] = 1 - colored_tiles[k]
                slide_name = bags_list[i].split(os.sep)[-1]
                color_map = transform.resize(color_map, (color_map.shape[0] * 32, color_map.shape[1] * 32), order=0)
                io.imsave(os.path.join(args.map_path, slide_name + '_{}.png'.format(c)), img_as_ubyte(color_map))

            print('save img:{}'.format(slide_name))

def get_feature(model, x):
    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.avgpool(x)
        x = x.view(x.size(0),-1)

    return x


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--predict_path', type=str, default=None, help='model weights')
    parser.add_argument('--bag_path', type=str, default='./dataset/TCGA/single/test/0')
    parser.add_argument('--patch_ext', type=str, default='jpeg')
    parser.add_argument('--map_path', type=str, default='out_put')
    parser.add_argument('--model_type', type=str, choices=['my_model', 'clam_model', 'single_model', 'single_model_sb'],
                        default='my_model', help='type of model (default: my_model)')
    parser.add_argument('--exp_code', type=str, default='no_smooth', help='experiment code for model')
    parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')
    args = parser.parse_args()

    if args.exp_code == 'no_smooth':
        args.predict_path = './results_final/712_split/TCGA/my_model/inst_only&control/weight_0.4&B_8/s_9_checkpoint_loss.pt'
        args.map_path = 'output_no_smooth_0'
    else:
        args.predict_path = './results_final/712_split/TCGA/my_model/all_model&control/weight_0.4&B_8&smmoth_15&gamma_1.5/s_9_checkpoint_loss.pt'
        args.map_path = 'output_all_model_0'

    model_dict = {"dropout": args.drop_out,
                  'n_classes': args.n_classes,
                  'subtyping':True,
                  }
    instance_loss_fn = nn.CrossEntropyLoss()

    model = My_model(**model_dict, instance_loss_fn=instance_loss_fn)
    model.relocate()

    pre_dict = torch.load(args.predict_path)
    model.load_state_dict(pre_dict, strict=True)

    resnet = models.resnet50()
    resnet.load_state_dict(torch.load('resnet50-19c8e357.pth'))
    resnet = resnet.cuda()

    bags_list = glob.glob(os.path.join(args.bag_path, '*'))
    os.makedirs(args.map_path, exist_ok=True)

    test(args, bags_list, model, resnet)