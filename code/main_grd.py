# System libs
import os
import random
import time

# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# Our libs
from arguments import ArgParser
from dataset import MUSICMixDataset
from models import ModelBuilder
from utils import AverageMeter, warpgrid,  makedirs
from viz import plot_grdloss
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, nets, crit):
        super(NetWrapper, self).__init__()
        self.net_sound_ground, self.net_frame_ground, self.net_grounding = nets
        self.maxpool = torch.nn.AdaptiveMaxPool1d(1)
        self.crit = crit
        self.cts = nn.CrossEntropyLoss()

    def forward(self, batch_data, args):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']
        mag_mix = mag_mix + 1e-10

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(
                warpgrid(B, 256, T, warp=True)).to(args.device)
            mag_mix = F.grid_sample(mag_mix, grid_warp)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp)

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                # for simplicity, mag_N > 0.5 * mag_mix
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)

        gt_masks[1] = torch.mul(gt_masks[3], 0.)
        gt_masks[3] = torch.mul(gt_masks[3], 0.)

        # LOG magnitude
        log_mag_mix = torch.log1p(mag_mix).detach()
        log_mag0 = torch.log1p(mags[0]).detach()
        log_mag2 = torch.log1p(mags[2]).detach()

        # grounding
        feat_sound_ground = self.net_sound_ground(log_mag_mix)

        feat_frames_ground = [None for n in range(N)]
        for n in range(N):
            feat_frames_ground[n] = self.net_frame_ground.forward_multiframe(frames[n])

        # Grounding for sep
        g_sep = [None for n in range(N)]
        x = [None for n in range(N)]
        for n in range(N):
            g_sep[n] = self.net_grounding(feat_sound_ground, feat_frames_ground[n])
            x[n] = torch.softmax(g_sep[n].clone(), dim=-1)

        # Grounding module
        g_pos = self.net_grounding(self.net_sound_ground(log_mag0) , feat_frames_ground[0])
        g_pos1 = self.net_grounding(self.net_sound_ground(log_mag0), feat_frames_ground[1])
        g_neg = self.net_grounding(self.net_sound_ground(log_mag2), feat_frames_ground[0])

        # Grounding for solo sound
        g_solo = [None for n in range(N)]
        g_solo[0] = self.net_grounding(self.net_sound_ground(log_mag0), feat_frames_ground[0])
        g_solo[1] = self.net_grounding(self.net_sound_ground(log_mag0), feat_frames_ground[1])
        g_solo[2] = self.net_grounding(self.net_sound_ground(log_mag2), feat_frames_ground[2])
        g_solo[3] = self.net_grounding(self.net_sound_ground(log_mag2), feat_frames_ground[3])
        for n in range(N):
            g_solo[n] = torch.softmax(g_solo[n], dim=-1)
        g = [torch.softmax(g_pos, dim=-1), torch.softmax(g_neg, dim=-1), x, g_solo]

        p = torch.zeros(B).cuda()
        n = torch.ones(B).cuda()

        cts_pos = torch.zeros(B).cuda()
        cts_pos1 = torch.zeros(B).cuda()
        for i in range(B):
            cts_pos[i] = self.cts(g_pos[i:i + 1], p[i:i + 1].long())
            cts_pos1[i] = self.cts(g_pos1[i:i + 1], p[i:i + 1].long())

        # 5. loss
        err = torch.min(cts_pos, cts_pos1).mean() + self.cts(g_neg, n.long()) \
              #+ torch.min(self.cts(g_sep[0], p.long()), self.cts(g_sep[1], p.long())) + torch.min(self.cts(g_sep[2], p.long()), self.cts(g_sep[3], p.long()))
        return err, g


def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=True)

    # switch to eval mode
    netWrapper.eval()
    loss_meter = AverageMeter()

    for i, batch_data in enumerate(loader):
        # forward pass
        err, g = netWrapper.forward(batch_data, args)
        err = err.mean()
        loss_meter.update(err.item())

        print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))
        grd_acc = np.sum(
            np.round(g[0][:, 0].detach().cpu().numpy()) + (np.round(g[1][:, 1].detach().cpu().numpy()))) / (
                          2 * len(np.round(g[0][:, 0].detach().cpu().numpy())))
        grd_mix_acc = (np.sum(
            np.round(g[2][0][:, 0].detach().cpu().numpy()) + np.round(g[2][1][:, 1].detach().cpu().numpy())
            + np.round(g[2][2][:, 0].detach().cpu().numpy()) + (
                np.round(g[2][3][:, 1].detach().cpu().numpy())))) / (
                              4 * len(np.round(g[2][0][:, 0].detach().cpu().numpy())))

        grd_solo_acc = (np.sum(
            np.round(g[3][0][:, 0].detach().cpu().numpy()) + np.round(g[3][1][:, 1].detach().cpu().numpy())
            + np.round(g[3][2][:, 0].detach().cpu().numpy()) + (
                np.round(g[3][3][:, 1].detach().cpu().numpy())))) / (
                               4 * len(np.round(g[3][0][:, 0].detach().cpu().numpy())))

        print('Grounding acc {:.2f}, Solo Grounding acc: {:.2f}, Sep Grounding acc: {:.2f}'.format(grd_acc, grd_solo_acc,
                                                                                                 grd_mix_acc))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_grdloss(args.ckpt, history)

# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        netWrapper.zero_grad()
        err,_ = netWrapper.forward(batch_data, args)
        err = err.mean()

        # backward
        err.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, lr_synthesizer: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, args.lr_synthesizer,
                          err.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())


def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound_ground, net_frame_ground, net_grounding) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_sound_ground.state_dict(),
               '{}/sound_ground_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_frame_ground.state_dict(),
               '{}/frame_ground_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_grounding.state_dict(),
               '{}/grounding_{}'.format(args.ckpt, suffix_latest))

    cur_err = history['val']['err'][-1]
    if cur_err < args.best_err:
        args.best_err = cur_err
        torch.save(net_sound_ground.state_dict(),
                       '{}/sound_ground_{}'.format(args.ckpt, suffix_best))
        torch.save(net_frame_ground.state_dict(),
                       '{}/frame_ground_{}'.format(args.ckpt, suffix_best))
        torch.save(net_grounding.state_dict(),
                       '{}/grounding_{}'.format(args.ckpt, suffix_best))


def create_optimizer(nets, args):
    (net_sound_ground, net_frame_ground, net_grounding) = nets
    param_groups = [{'params': net_sound_ground.parameters(), 'lr': args.lr_sound_ground},
                    {'params': net_grounding.parameters(), 'lr': args.lr_grounding}]
    return torch.optim.Adam(param_groups)


def adjust_learning_rate(optimizer, args):
    args.lr_sound_ground *= 0.1
    args.lr_grounding *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


def main(args):
    # Network Builders
    builder = ModelBuilder()
    net_sound_ground = builder.build_sound_ground(
        arch=args.arch_sound_ground,
        weights=args.weights_sound_ground)
    net_frame_ground = builder.build_frame_ground(
        arch=args.arch_frame_ground,
        pool_type=args.img_pool,
        weights=args.weights_frame_ground)
    net_grounding = builder.build_grounding(
        arch=args.arch_grounding,
        weights=args.weights_grounding)
    nets = (net_sound_ground, net_frame_ground, net_grounding)
    crit = builder.build_criterion(arch=args.loss)

    # Dataset and Loader
    dataset_train = MUSICMixDataset(
        args.list_train, args, split='train')
    dataset_val = MUSICMixDataset(
        args.list_val, args, max_sample=args.num_val, split=args.split)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False)
    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # Wrap networks
    netWrapper = NetWrapper(nets, crit)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    # Set up optimizer
    optimizer = create_optimizer(nets, args)

    # History of performance
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': []}}

    # Eval mode
    if args.mode == 'eval':
        evaluate(netWrapper, loader_val, history, 0, args)
        print('Evaluation Done!')
        return

    # Training loop
    for epoch in range(1, args.num_epoch + 1):
        train(netWrapper, loader_train, optimizer, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(netWrapper, loader_val, history, epoch, args)
            # checkpointing
            checkpoint(nets, history, epoch, args)

        # drop learning rate
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)

    print('Training Done!')


if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.device = torch.device("cuda")

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}-{}'.format(
            args.arch_frame, args.arch_sound, args.arch_synthesizer)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        args.id += '-{}'.format(args.img_pool)
        if args.binary_mask:
            assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-channels{}'.format(args.num_channels)
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    args.vis = os.path.join(args.ckpt, 'visualization/')
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
    elif args.mode == 'eval':
        args.weights_frame_ground = os.path.join(args.ckpt, 'frame_ground_best.pth')
        args.weights_sound_ground = os.path.join(args.ckpt, 'sound_ground_best.pth')
        args.weights_grounding = os.path.join(args.ckpt, 'grounding_best.pth')

    # initialize best error with a big number
    args.best_err = float("inf")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
