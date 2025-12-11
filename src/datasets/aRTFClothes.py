import glob
import json
import os
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class aRTFClothesDataset(Dataset):
	"""Dataset loader for the aRTFClothes collection.

	The dataset layout is similar to ViCoS but only provides RGB frames and grasping
	point annotations. Background folders follow the pattern ``bg=trainlocation_*`` or
	``bg=testlocation_*`` which we use for the default train/test split.
	"""

	def __init__(
		self,
		root_dir: str,
		split: str = 'train',
		subfolders: Optional[Iterable] = None,
		fixed_bbox_size: int = 15,
		resize_factor: Optional[float] = None,
		transform=None,
		transform_only_valid_centers: bool = False,
		transform_per_sample_rng: bool = False,
		MAX_NUM_CENTERS: int = 1024,
		valid_img_names: Optional[List[str]] = None,
		use_depth: bool = False,
		segment_cloth: bool = False,
		num_cpu_threads: int = 1,
		**kwargs,
	):
		if use_depth:
			raise ValueError("aRTFClothes dataset does not provide depth maps.")
		if segment_cloth:
			raise ValueError("aRTFClothes dataset does not provide segmentation masks.")

		if num_cpu_threads:
			torch.set_num_threads(num_cpu_threads)

		self.root_dir = os.path.abspath(root_dir)
		self.split = (split or kwargs.get('type', 'train')).lower()
		if self.split not in {'train', 'test', 'all'}:
			raise ValueError("split/type must be one of {'train', 'test', 'all'}.")

		self.fixed_bbox_size = fixed_bbox_size
		self.resize_factor = resize_factor
		self.MAX_NUM_CENTERS = MAX_NUM_CENTERS

		self.transform = transform
		self.transform_only_valid_centers = transform_only_valid_centers
		self.transform_per_sample_rng = transform_per_sample_rng
		self.rng = np.random.default_rng(1337)

		self.annotations = self._load_annotations()

		if subfolders is None:
			subfolders = self._discover_subfolders()

		image_list = self._collect_images(subfolders)

		if valid_img_names is not None:
			def filter_by_name(path):
				return any(token in path for token in valid_img_names)
			image_list = list(filter(filter_by_name, image_list))

		self.image_list = image_list
		self.size = len(self.image_list)
		print(f'aRTFClothesDataset[{self.split}] of size {self.size}')

	def __len__(self):
		return self.size

	def __getitem__(self, index: int):
		im_fn = self.image_list[index]
		fn = os.path.splitext(os.path.basename(im_fn))[0]

		image = Image.open(im_fn)
		org_im_size = image.size
		im_size = org_im_size

		if self.resize_factor is not None and self.resize_factor != 1.0:
			im_size = (int(image.size[0] * self.resize_factor), int(image.size[1] * self.resize_factor))
			image = image.resize(im_size, Image.BILINEAR)

		sample = dict(
			image=image,
			im_name=im_fn,
			org_im_size=np.array(org_im_size),
			im_size=im_size,
			index=index,
		)

		label = torch.zeros((im_size[1], im_size[0]), dtype=torch.uint8)
		instances = torch.zeros((im_size[1], im_size[0]), dtype=torch.int16)
		orientation = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.float32)

		ann = self.annotations.get(os.path.abspath(im_fn), dict())
		points = ann.get('points', [])

		centers = []
		if points:
			M = self.fixed_bbox_size
			instance_counter = 1

			for x1, y1, x2, y2 in points:
				pt1 = np.array([x1, y1], dtype=np.float32)
				pt2 = np.array([x2, y2], dtype=np.float32)

				if self.resize_factor is not None and self.resize_factor != 1.0:
					pt1 *= self.resize_factor
					pt2 *= self.resize_factor

				direction = pt1 - pt2
				angle = float(np.arctan2(direction[0], direction[1]))

				i0, j0 = int(pt1[0]), int(pt1[1])
				i1, i2 = max(0, i0 - M), min(im_size[0], i0 + M)
				j1, j2 = max(0, j0 - M), min(im_size[1], j0 + M)

				if i1 >= i2 or j1 >= j2:
					continue

				orientation[:, j1:j2, i1:i2] = angle
				label[j1:j2, i1:i2] = 1
				instances[j1:j2, i1:i2] = instance_counter
				centers.append(pt1)

				instance_counter += 1

		center_array = np.zeros((self.MAX_NUM_CENTERS, 2), dtype=np.float32)
		if centers:
			centers = np.array(centers, dtype=np.float32)
			count = min(len(centers), self.MAX_NUM_CENTERS)
			center_array[:count] = centers[:count]
			original_center_count = len(centers)
		else:
			original_center_count = 0

		sample['center'] = center_array
		sample['label'] = label.unsqueeze(0)
		sample['mask'] = (label > 0).unsqueeze(0)
		sample['orientation'] = orientation
		sample['instance'] = instances.unsqueeze(0)
		sample['ignore'] = torch.zeros((1, im_size[1], im_size[0]), dtype=torch.uint8)
		sample['name'] = im_fn

		if self.transform is not None:
			import copy
			do_transform = True
			attempts = 0

			while do_transform:
				if attempts > 0 and attempts % 10 == 0:
					print(f"WARNING: unable to generate valid transform for {attempts} iterations (index={index})")
				rng = self.rng if not self.transform_per_sample_rng else np.random.default_rng(1337)
				new_sample = self.transform(copy.deepcopy(sample), rng)

				if not self.transform_only_valid_centers or original_center_count == 0:
					do_transform = False
					sample = new_sample
				else:
					centers_np = new_sample['center']
					valid = (centers_np[:, 0] > 0) | (centers_np[:, 1] > 0)
					num_centers = int(valid.sum())
					if isinstance(self.transform_only_valid_centers, bool):
						do_transform = num_centers == 0
					elif isinstance(self.transform_only_valid_centers, int):
						do_transform = num_centers < self.transform_only_valid_centers
					elif isinstance(self.transform_only_valid_centers, float):
						target = max(1, int(original_center_count * self.transform_only_valid_centers))
						do_transform = num_centers < target
					else:
						do_transform = False

					if not do_transform:
						sample = new_sample
				attempts += 1

		return sample

	def _load_annotations(self):
		annot_path = os.path.join(self.root_dir, 'annotations.json')
		if not os.path.exists(annot_path):
			raise FileNotFoundError(f'annotations.json not found in {self.root_dir}')

		with open(annot_path, 'r') as f:
			annotations = json.load(f)

		ann_abs = {}
		for rel_path, value in annotations.items():
			full_path = os.path.abspath(os.path.join(self.root_dir, rel_path))
			ann_abs[full_path] = value

		return ann_abs

	def _discover_subfolders(self):
		bg_dirs = sorted(glob.glob(os.path.join(self.root_dir, 'bg=*')))
		if self.split != 'all':
			prefix = 'bg=trainlocation_' if self.split == 'train' else 'bg=testlocation_'
			bg_dirs = [bg for bg in bg_dirs if os.path.basename(bg).startswith(prefix)]

		subfolders = []
		for bg in bg_dirs:
			cloth_dirs = sorted(glob.glob(os.path.join(bg, 'cloth=*')))
			data_subfolders = [os.path.basename(cd) for cd in cloth_dirs]
			if not data_subfolders:
				continue
			subfolders.append(dict(folder=os.path.basename(bg), data_subfolders=data_subfolders))

		return subfolders

	def _collect_images(self, subfolders: Iterable):
		image_list: List[str] = []
		for sub in subfolders:
			if isinstance(sub, dict):
				bg_folder = os.path.join(self.root_dir, sub['folder'])
				for data_path in sub.get('data_subfolders', []):
					rgb_path = os.path.join(bg_folder, data_path, 'rgb')
					image_list.extend(sorted(glob.glob(os.path.join(rgb_path, '*'))))
			else:
				bg_folder = os.path.join(self.root_dir, sub)
				rgb_path = os.path.join(bg_folder, 'rgb')
				if os.path.isdir(rgb_path):
					image_list.extend(sorted(glob.glob(os.path.join(rgb_path, '*'))))
				else:
					for cloth_dir in sorted(glob.glob(os.path.join(bg_folder, 'cloth=*'))):
						image_list.extend(sorted(glob.glob(os.path.join(cloth_dir, 'rgb', '*'))))
		return image_list
