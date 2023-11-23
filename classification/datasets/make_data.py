import torch
import os
from torchvision import datasets, transforms


__imagenet_mean_std = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class KeyError(Exception):
	def __init__(self, msg):
		self.msg = msg

	def __str__(self):
		return self.msg

def mobilenetv2_train_preproccess(image_size, normalize=__imagenet_mean_std):
	train_transforms = transforms.Compose([
		transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(**normalize),
		])
	return train_transforms

def mobilenetv2_test_preproccess(image_size, normalize=__imagenet_mean_std):
	scale_size = int(image_size / 0.875)
	test_transforms = transforms.Compose([
		transforms.Resize(scale_size),
		transforms.CenterCrop(image_size),
		transforms.ToTensor(),
		transforms.Normalize(**normalize),
		])
	return test_transforms

def get_transforms(mode, image_size):
	if mode == 'train':
		train_transforms = mobilenetv2_train_preproccess(image_size=224)
		return train_transforms
	elif mode == 'val':
		test_transforms = mobilenetv2_test_preproccess(image_size=224)
		return test_transforms
	else:
		raise KeyError('mode must be train or val')

def make_dataloader(dataroot, train_batch_size, test_batch_size, dist, num_tasks, global_rank, workers=4, image_size=224):
	train_dataset = datasets.ImageFolder(root=os.path.join(dataroot, 'train'), transform=get_transforms(mode='train', image_size=image_size))
	val_dataset = datasets.ImageFolder(root=os.path.join(dataroot, 'val'), transform=get_transforms(mode='val', image_size=image_size))
	if dist:
		train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
	else:
		train_sampler = torch.utils.data.RandomSampler(train_dataset)
	train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, num_workers=workers, \
		pin_memory=True, drop_last=True,)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=workers, \
		pin_memory=True, drop_last=False)
	return train_loader, val_loader