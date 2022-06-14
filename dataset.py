"""
Preprocess dataset from txt files.
author: phatnt
date: Aug-10-2021
"""

import os 
import cv2
import torch
import numpy as np

IMAGE_EXTS = ['jpg', 'png', 'bmp', 'jpeg']

class CustomDataset(torch.utils.data.Dataset):
	"""
	Customize your own Dataset to Pytorch-type dataset.
	"""
	def __init__(self, txt_file, transform=None):
		"""
		Args:
			- txt_file: path to txt_file ("image_path", id)
			- transform: Transforms type(Resize, Normalize, To_Tensor)
		"""
		assert os.path.isfile(txt_file)
		self.image_paths = []
		self.targets = []
		self.transform = transform

		with open(txt_file, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		for line in lines:
			line = line.strip()
			image_path, target = line.split("\"")[1], float(line.split("\"")[2].strip())
			self.image_paths.append(image_path)
			self.targets.append(target)
	
	def __len__(self):
		return len(self.image_paths)
	
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		assert os.path.isfile(self.image_paths[idx]), f'[ERROR] Could not found {self.image_paths[idx]}!'
		exts = self.image_paths[idx].split('.')[-1]
		if exts in IMAGE_EXTS:
			image = cv2.imread(self.image_paths[idx])
			assert image is not None, '[ERROR] image is None'
		elif exts == 'npy':
			image = np.load(self.image_paths[idx])
			image = np.squeeze(image)
		else:
			raise Exception(f'[ERROR] Unsupported extension: {exts}')
		target = float(self.targets[idx])
		path = self.image_paths[idx]
		sample = {'image': image, 'path': path, 'target': target}
		if self.transform:
			sample = self.transform(sample)
		return sample

class Resize(object):
	def __init__(self, imgsz):
		assert isinstance(imgsz, int)
		self.imgsz = imgsz
		
	def __call__(self, sample):
		image, path, target = sample['image'], sample['path'], sample['target']
		resized = cv2.resize(image, (self.imgsz, self.imgsz), interpolation = cv2.INTER_AREA)
		return {'image': resized, 'path': path, 'target': target}
	
class Normalize(object):
	def __call__(self, sample):
		image, path, target = sample['image'], sample['path'], sample['target']
		image = np.float32(image)
		image = image*(1/255)
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		image = (image - mean) / std
		return {'image': image, 'path': path, 'target': target}

class ToTensor(object):
	def __call__(self, sample):
		image, path, target = sample['image'], sample['path'], sample['target']
		image = image.transpose((2, 0, 1))
		image = np.asarray(image).astype(np.float32)
		return {'image': torch.from_numpy(image),
				'path': path,
				'target': target}