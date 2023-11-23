import torch
import time
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange
from torch.cuda.amp import GradScaler,autocast
from .util import ProgressMeter, reduce_value, get_rank

def train(model, train_loader, epoch, optimizer, criterion, device, log_interval, using_amp, scheduler, dist, clip_gradient, max_norm):
    start = time.time()
    model.train()
    batch_time = AverageMeter('Batch Time', ':6.3f')
    epoch_time = AverageMeter('Epoch Time', ':6.3f')
    data_time  = AverageMeter('Data Time', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    scaler = GradScaler()
    end = time.time()

    if get_rank():
        train_loader = tqdm(train_loader, leave=False, desc='training')

    for batch_idx, (data, target) in enumerate(train_loader):
        scheduler.step()
        data_time.update(time.time() - end)
        data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
        optimizer.zero_grad()
        if using_amp:
            # Using fp16 
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            if clip_gradient:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Using fp32
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if clip_gradient:
                clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if dist:
            loss = reduce_value(loss.data)
            acc1 = reduce_value(acc1)
            acc5 = reduce_value(acc5)
        else:
            loss = loss.data
        loss_meter.update(loss.item(), data.size(0))
        acc1_meter.update(acc1[0], data.size(0))
        acc5_meter.update(acc5[0], data.size(0))

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx != 0 and batch_idx % log_interval == 0 and get_rank():
            tqdm.write(
                f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)}({100. * batch_idx / len(train_loader):.0f}%)]. '
                f'Top-1 accuracy: {acc1_meter.avg:.4f}%. '
                f'Top-5 accuracy: {acc5_meter.avg:.4f}%. '
                f'Loss: {loss_meter.avg:.4f}. '
                f'Data time: {data_time.avg:.5f}. '
                f'Batch time: {batch_time.avg:.5f}. '
                )
    epoch_time.update(time.time() - start)
    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg, epoch_time.avg

def test(model, val_loader, criterion, device, log_interval, dist):
    model.eval()
    start = time.time()
    epoch_time = AverageMeter('Epoch Time', ':6.4f')
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [loss_meter, acc1_meter, acc5_meter], prefix='Test: ')

    if get_rank():
        val_loader = tqdm(val_loader, leave=False, desc='evaluating')

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if dist:
                loss = reduce_value(loss.data)
                acc1 = reduce_value(acc1)
                acc5 = reduce_value(acc5)
            else:
                loss = loss.data
            loss_meter.update(loss.item(), data.size(0))
            acc1_meter.update(acc1[0], data.size(0))
            acc5_meter.update(acc5[0], data.size(0))

            if batch_idx != 0 and  batch_idx % log_interval == 0 and get_rank():
                progress.display(batch_idx)

    epoch_time.update(time.time() - start)
    if get_rank():
        tqdm.write(
            f'Test set: Epoch inference time: {epoch_time.avg:.4f}, Average loss: {loss_meter.avg:.4f}, Top1: {acc1_meter.avg:.4f}%, Top5: {acc5_meter.avg:.4f}%'
            )
    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg, epoch_time.avg

def finaltest(model, val_loader, criterion, device, log_interval, dist):
    model.eval()
    start = time.time()
    epoch_time = AverageMeter('Epoch Time', ':6.4f')
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [loss_meter, acc1_meter, acc5_meter], prefix='Test: ')
    val_loader = tqdm(val_loader, leave=False, desc='evaluating')

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if dist:
                loss = reduce_value(loss.data)
                acc1 = reduce_value(acc1)
                acc5 = reduce_value(acc5)
            else:
                loss = loss.data
            loss_meter.update(loss.item(), data.size(0))
            acc1_meter.update(acc1[0], data.size(0))
            acc5_meter.update(acc5[0], data.size(0))

            if batch_idx != 0 and  batch_idx % log_interval == 0 and get_rank():
                progress.display(batch_idx)

    epoch_time.update(time.time() - start)
    tqdm.write(
            f'Test set: Epoch inference time: {epoch_time.avg:.4f}, Average loss: {loss_meter.avg:.4f}, Top1: {acc1_meter.avg:.4f}%, Top5: {acc5_meter.avg:.4f}%'
            )
    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg, epoch_time.avg

def accuracy(outputs, targets, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)