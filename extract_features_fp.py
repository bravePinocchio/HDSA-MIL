import os
gpu_index = (0,2)
gpu_ids = tuple(gpu_index)
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)
import torch
import torch.nn as nn
import numpy as np
import os, glob, time
import torchvision.transforms.functional as VF
import pandas as pd
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from my_test import bagdataset
import h5py
from torchvision import transforms
import openslide

device = torch.device('cuda'
					  ) if torch.cuda.is_available() else torch.device('cpu')
class bag_lbp_dataset(Dataset):
    def __init__(self, root = 'Desktop/screenshots/', pretrained = True):
        self.root = root
        self.raw_samples = glob.glob(self.root + '/*')
        self.trans = self.get_trans(pretrained)

    def get_trans(self, pretrained):
        if pretrained:
            mean = (0.7219808, 0.58108766, 0.70086701)
            std = (0.18555619, 0.2276182, 0.18211643)
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)

        trans = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        return trans

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):

        patch_dir = self.raw_samples[idx]

        image = Image.open(patch_dir)  ## PIL image
        image = self.trans(image)

        return image

def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file),coords 的 .h5 文件
		output_path: directory to save computed features (.h5 file)
		wsi: 当前slide 的 open_slide的对象
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			features, _ = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

def extract_features_fp():
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	model = resnet50_baseline(pretrained=True)
	model = model.to(device)
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.eval()
	total = len(bags_dataset)
	for bag_candidate_idx in range(total):
		slide_name = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		slide_id = bags_dataset[bag_candidate_idx].split('.')[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_name + args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(h5_file_path, output_path, wsi,
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20,
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))

def compute_my_patch(base_path, args):
	print('loading model checkpoint')
	model = resnet50_baseline(pretrained=True)

	model = model.to(device)

	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.eval()
	all_path = glob.glob(base_path + '/*/*')
	save_path = os.path.join('./dataset/BARCS/feature', args.type)
	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	for path in all_path:
		print(path)
		type = path.split('/')[-2]
		slide_id = path.split('/')[-1]
		dataset = bagdataset(path)
		loader = DataLoader(dataset=dataset, batch_size= 128)
		feats = []
		low_feats = []
		x_list = []
		for i, batch in enumerate(loader):
			with torch.no_grad():
				if i % 20 == 0:
					print('batch {}/{}, {} files processed'.format(i, len(loader), i * 128))
				batch = batch.to(device, non_blocking=True)  # [256, 3, 224, 224]
				feature, low_feature  = model(batch)
				low_feats.append(low_feature)
				feats.append(feature)
		feats = torch.cat(feats)
		# low_feats = low_feats.cpu()
		feats = feats.cpu()
		torch.save(feats, os.path.join(save_path, type, slide_id + '.pt'))

def compute_my_pyramid_patch():
	base_path = './dataset/TCGA/pyramid/train/1'
	slides = glob.glob(base_path + '/*')
	out_path = 'my_data/class_feature/pt_feature_pyramid'
	print('loading model checkpoint')
	model = resnet50_baseline(pretrained=True)
	model = model.to(device)
	model.eval()
	feats = []
	for slide in slides:
		slide_id = slide.split('/')[-1]
		print('process slide_id:{}'.format(slide_id))
		low_images = glob.glob(slide + '/*.jpeg') + glob.glob(slide + '/*.jpg')
		for low_img in low_images:
			low_feats = Image.open(low_img)
			low_feats = VF.to_tensor(low_feats).float().cuda()
			low_feats = low_feats.unsqueeze(0)
			low_feats = model(low_feats)
			low_feats = low_feats.detach().cpu().numpy()
			high_feats = np.zeros(1024)
			high_images =  glob.glob(low_img.replace('.jpeg', os.sep + '*.jpeg'))
			if len(high_images)==0:
				feats.append(low_feats)
			else:
				for high_img in high_images:
					img = Image.open(high_img)
					img = VF.to_tensor(img).float().cuda()
					img = img.unsqueeze(0)
					img_feats = model(img)
					img_feats = img_feats * (1/len(high_images))
					img_feats = img_feats.detach().cpu().numpy()
					high_feats = high_feats + img_feats
				low_feats = 0.6 * low_feats + 0.4 * high_feats
				feats.append(low_feats)

		torch.save(feats, os.path.join(out_path,  slide_id + '.pt'))





if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='single patch feature extracting')
	parser.add_argument('--type', type=str, default='val', help='data csv path')
	args = parser.parse_args()
	#extract_features_fp()

	#extract_mean()
	base_path = os.path.join('./dataset/BARCS/single',args.type)
	compute_my_patch(base_path, args)

