import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from model import VisionTransformer, subsequent_mask, make_model
from config import get_train_config
#from checkpoint import load_checkpoint
from utils import setup_device, accuracy, MetricTracker, TensorboardWriter
import os
from tensorboardX import SummaryWriter
from yl360Dataset import IQADataset
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train_epoch(tensor_writer, config, epoch, model, data_loader, tgt_mask, criterion, optimizer, lr_scheduler, metrics,  device_ids1, device=torch.device('cpu')):
    #metrics.reset()

    # training loop
    for batch_idx, (batch_data, batch_target, batch_gt) in enumerate(data_loader):
        #batch_data = batch_data.to(device)
        #batch_target = batch_target.to(device)

        optimizer.zero_grad()
        batch_pred = model.forward(batch_data, batch_target, tgt_mask)
        loss = criterion(batch_pred, batch_gt)
        loss.to('cuda')
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))
        #
        # metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        # metrics.update('loss', loss.item())
        # metrics.update('acc1', acc1.item())
        # metrics.update('acc5', acc5.item())
        tensor_writer.add_scalar('Train/Iter/Loss', loss.item(), batch_idx)
        save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids1, best=False)
        #save_model(config.checkpoint_dir)


        if batch_idx % 1 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} "
                    .format(epoch, batch_idx, len(data_loader), loss.item()))
    #return metrics.result()


def valid_epoch(epoch, model, data_loader, criterion, metrics, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    # validation loop
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)

            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
            acc1, acc5 = accuracy(batch_pred, batch_target, topk=(1, 5))

            losses.append(loss.item())
            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

    loss = np.mean(losses)
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', acc1)
    metrics.update('acc5', acc5)
    return metrics.result()


def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str(save_dir + 'current.pth')
    torch.save(state, filename)

    if best:
        filename = str(save_dir + 'best.pth')
        torch.save(state, filename)


def get_data_loaders(config, train_batch_size):

    train_dataset = IQADataset(config, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=0)

    return train_loader


def main():
    torch.set_num_threads(12)
    torch.manual_seed(10021)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(10000)
    #random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    # Set starting epoch
    start_epoch = 0

    # Set configuration
    config = get_train_config()

    # device
    device, device_ids = setup_device(config.n_gpu)

    # tensorboard
    writer = SummaryWriter(log_dir=config.summary_dir)
    #writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']
    train_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)
    #valid_metrics = MetricTracker(*[metric for metric in metric_names], writer=writer)

    # create model
    print("create model")
    # x = torch.randn((4, 3, 512, 512)).float().cuda()
    # y = torch.randn((4, 250, 2)).float().cuda()

    # print(x)
    tgt_mask = subsequent_mask(config.num_viewport)
    tgt_mask = tgt_mask.float().cuda()
    model = make_model(2, config.num_decLayer)  #
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
        momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        pct_start=config.warmup_steps / config.train_steps,
        total_steps=config.train_steps)

    #load checkpoint
    if config.checkpoint_path:
        #  想加载哪个pth文件 通过外部程序写入
        checkpoint = torch.load(config.checkpoint_path)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("Load pretrained weights from {}".format(config.checkpoint_path))

    # send model to device
    model = model.to('cuda')
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    print("create dataloaders")
    train_dataloader = get_data_loaders(config, config.batch_size)



    # train_dataloader = eval("{}DataLoader".format(config.dataset))(
    #                 data_dir=os.path.join(config.data_dir, config.dataset),
    #                 image_size=config.image_size,
    #                 batch_size=config.batch_size,
    #                 num_workers=config.num_workers,
    #                 split='train')

    # training criterion
    print("create criterion and optimizer")
    criterion = nn.MSELoss(reduce= True) ############################ MSE Loss ##########################################

    # create optimizers and learning rate scheduler


    # start training
    print("start training")
    best_acc = 0.0
    epochs = config.train_steps // len(train_dataloader)
    for epoch in range(start_epoch+1, epochs + 1):
        log = {'epoch': epoch}

        # train the model
        model.train()
        train_epoch(writer, config, epoch, model, train_dataloader, tgt_mask, criterion, optimizer, lr_scheduler, train_metrics, device, device_ids)
        # save model
        #save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False)
        if epoch % 1 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }
            filename = os.path.join(config.checkpoint_dir, "model_opti_lr_state.epoch{}.pth".format(epoch))
            torch.save(state, filename)
            #torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, "model_opti_lr_state.epoch{}.pth".format(epoch)))
            print('Successfully saved model of EPOCH{}'.format(epoch))

        # print logged informations to the screen
        for key, value in log.items():
            print('    {:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    main()
    # print(subsequent_mask(10))
    # print(np.triu(np.ones((1,10,10)), k=1).astype('uint8'))
    # print(-1e9)
    # a=torch.zeros((10,10))
    # a.masked_fill(subsequent_mask(10) == 0, 1)
    # print(a.masked_fill(subsequent_mask(10) == 0, 1))




















