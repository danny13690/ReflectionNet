import os

import torch
import pandas as pd
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from xml_processor import XML_preprocessor


class Stanford40Dataset(Dataset):
	def __init__(self, data_path, mode, transform=None):
		self.data_path = data_path
		self.transform = transform
		self.mode = mode

		self.image_names = []

		xml_path = os.path.join(data_path,"XMLAnnotations/")
		xml_processor = XML_preprocessor(xml_path)
		self.labels_dict = xml_processor.data

		if mode == "train":
			splits_path = os.path.join(data_path,"ImageSplits","train.txt")
			with open(splits_path, 'r') as file:
				for line in file:
					line = line.strip()
					self.image_names.append(line)

		elif mode == "test":
			splits_path = os.path.join(data_path,"ImageSplits","test.txt")
			with open(splits_path, 'r') as file:
				for line in file:
					line = line.strip()
					self.image_names.append(line)


	def __len__(self):
		return len(self.image_names)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		jpeg_path = os.path.join(self.data_path,"JPEGImages")
		image_name = self.image_names[idx]

		image_path = os.path.join(jpeg_path,image_name)
		image = Image.open(image_path)
		label = self.labels_dict[image_name]

		if self.transform != None:
			image = self.transform(image)

		return image, label
