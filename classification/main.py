import os
import torch
import logging
import warnings
import torchvision
import tempfile
import pathlib
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.distributed as dist
from configs import make_cfg_args
from datasets import make_dataloader
from networks.backbones import create_net
from utils import *
import warnings
warnings.filterwarnings('ignore')


def main(cfg, args, best_prec1):
	##################################
	# seed and dist init
	##################################
	init_distributed_mode(args)
	device = torch.device(args.device)
	seed = args.seed + get_rank()
	setup_seed(seed)
	torch.backends.cudnn.benchmark = True

	##################################
	# Logging setting
	##################################
	os.makedirs(cfg.record_dir, exist_ok=True)
	logging.basicConfig(
						filename=os.path.join(cfg.record_dir, cfg.run + '_' + str(args.seed) + '.log'),
						filemode='w',
						format='%(asctime)s: %(message)s',
						level=logging.INFO)
	warnings.filterwarnings("ignore")

	##################################
	# Load sampler dataset
	##################################
	train_loader, validate_loader = make_dataloader(dataroot=cfg.dataroot, train_batch_size=cfg.train.batch_size, \
		test_batch_size=cfg.test.batch_size, workers=cfg.workers, dist=args.distributed, num_tasks=get_world_size(), \
		global_rank=get_rank())

	##################################
	# Load model
	##################################
	model = create_net(args, cfg.attentions)
	flops, params = get_model_complexity_info(model, (224, 224), as_strings=False, print_per_layer_stat=False)
	model.to(device)
	if args.distributed:
		checkpoint_path = os.path.join(tempfile.gettempdir(), 'initial_weights.pt')
		if args.rank == 0:
			torch.save(model.state_dict(), checkpoint_path)
		dist.barrier()
		model.load_state_dict(torch.load(checkpoint_path, map_location=device))
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
	

	##################################
	# tensorboard setting
	##################################
	writer_path = pathlib.Path(cfg.tensor_Board, cfg.arch, cfg.attention_type, cfg.run + '_' + str(args.seed))
	writer_path.mkdir(parents=True, exist_ok=True)
	Writer = SummaryWriter(log_dir=writer_path)

	##################################
	# Load optimizer, scheduler, loss
	##################################
	criterion = CrossEntropyLabelSmooth().cuda()
	optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr*float(cfg.train.batch_size*args.world_size)/256., \
		momentum=cfg.train.momentum, weight_decay=cfg.train.weight_decay)
	scheduler = CosineAnnealingLR(optimizer, T_max=(cfg.train.epoch-args.warm)*len(train_loader), warmup='linear', \
		warmup_iters=args.warm*len(train_loader), eta_min=1e-8)

	##################################
	# Logging title
	##################################
	logging.info('-------------------------------config--------------------------------')
	logging.info(cfg)
	logging.info('-------------------------network information-------------------------')
	logging.info('Flops:  {:.3f}GMac  Params: {:.2f}M'.format(flops / 1e9, params / 1e6))
	logging.info('Network weights save to {}'.format(cfg.record_dir))
	logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.format(cfg.train.epoch, \
		cfg.train.batch_size, cfg.imagenet_train_size, cfg.imagenet_test_size))
	logging.info('-------------------------------model--------------------------------')
	logging.info(model)
	logging.info('--------------------------training progress--------------------------')

	##################################
	# Resume
	##################################
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			if 'optimizer' in checkpoint:
				optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	##################################
	# Start trainging
	##################################
	for epoch in range(0, cfg.train.epoch):
		if args.distributed:
			train_loader.sampler.set_epoch(epoch)
		tr_loss, tr_correct_1, tr_correct_5, train_epoch_time = train(model, train_loader, epoch, optimizer, \
			criterion, device, cfg.train.log_interval, cfg.using_amp, scheduler, args.distributed, cfg.train.clip_gradient, cfg.train.max_norm)
		logging.info(f'Training: Top-1 accuracy: {tr_correct_1:.4f}%. Top-5 accuracy: {tr_correct_5:.4f}%. Loss: {tr_loss:.4f}. Train epoch time: {train_epoch_time:.4f}')
		te_loss, te_correct_1, te_correct_5, inference_epoch_time = test(model, validate_loader, criterion, \
			device, cfg.test.log_interval, args.distributed)
		logging.info(f'Testing: Top-1 accuracy: {te_correct_1:.4f}%. Top-5 accuracy: {te_correct_5:.4f}%. Loss: {te_loss:.4f}. Inference epoch time: {inference_epoch_time:.4f}')
		logging.info('Epoch {:03d}, Learning Rate {:g}\n'.format(epoch, optimizer.param_groups[0]['lr']))
		Writer.add_scalar('train_loss',  tr_loss,      epoch)
		Writer.add_scalar('test_loss' ,  te_loss,      epoch)
		Writer.add_scalar('train_acc1',  tr_correct_1, epoch)
		Writer.add_scalar('test_acc1' ,  te_correct_1, epoch)

		##################################
		# Check point
		#################################
		is_best = te_correct_1 > best_prec1
		best_prec1 = max(te_correct_1, best_prec1)
		if get_rank() == 0:
			save_checkpoint({
				'epoch': epoch+1,
				'arch': args.arch,
				'state_dict': model.module.state_dict(),
				'best_prec1': best_prec1,
				'optimizer' : optimizer.state_dict(),
			}, is_best, cfg, args)
	if args.rank == 0:
		if os.path.exists(checkpoint_path) is True:
			os.remove(checkpoint_path)

	##################################
	# End trainging
	#################################
	logging.info('================= END =================')
	logging.info(f'Best Accuracy {best_prec1 :.4f}%')
	Writer.close()
	dist.destroy_process_group()


if __name__ == '__main__':
	cfg, args = make_cfg_args()
	main(cfg, args, best_prec1=0)