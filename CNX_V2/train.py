import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from utils.data_loader import ImageNet100Dataset
from utils.data_manger import ImageNet100
from utils.utils import Logger, AverageMeter, save_checkpoint
import utils.transform as T
from model.convNeXt import convnext_base
from model.convNeXt_CA import convnextca_base
from model.convNeXt_SA import convnextsa_base
from model.convNeXt_CASA import convnextcasa_base

parser = argparse.ArgumentParser(description='Train ConNeXtSa with cross entropy loss')

parser.add_argument('--root', default='E://self_dataset/imagenet100/imagenet100/', type=str, help='dataset path')
parser.add_argument('--height', default=224, type=int, help='height of an image')
parser.add_argument('--width', default=224, type=int, help='width of an image')

parser.add_argument('--max-epoch', default=4, type=int, help='maximum epoches to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--freeze-epoch', default=2, type=int, help='epoch of training with freeze')
parser.add_argument('--train-batch', default=32, type=int, help='train batch size with freeze')
parser.add_argument('--val-batch', default=32, help='val batch size')
parser.add_argument('--model', default="init", type=str, help='basic model backbone')

parser.add_argument('--lr-one', default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--lr-two', default=4e-4, type=float, help='initial learning rate')
parser.add_argument('--step-size', default=10, help='step size to decay learning rate(>0 mean this is enabled)')
parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay')
parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default:5e - 4)')
parser.add_argument('--num-workers', default=8, type=int, help='maximum number dataloader workers')
parser.add_argument('--print-freq', default=10, type=int, help='print logs')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--resume', default='', metavar='PATH', help='from this path restart training')
parser.add_argument('--weights',
                    default="D://PyCharm/xiaozhijie/modify_project/CNXA/CNX_X/weights/convnext_base_1k_224_ema.pth",
                    type=str, help='initial weight path')
parser.add_argument('--freeze-layers', default=True, type=bool, help='freeze same of layers in weights ')
parser.add_argument('--save-dir', default='logs/log', type=str, help='save training log and trained model')
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--device', default='cuda:0', help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--eval-step', default=2, type=int,
                    help='run evaluation for every N epochs (set to -1 val after training)')
parser.add_argument('--start-eval', default=0, type=int, help='start to evaluate after specific epoch')

args = parser.parse_args()


def train_one_eopch(epoch, model_train, model, criterion_class, optimizer, train_loader, use_gpu):
    # 打印信息
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model_train.train()

    end = time.time()
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        if use_gpu:
            imgs, labels = imgs.cuda(), labels.cuda()

        data_time.update(time.time() - end)  # 更新时间
        outputs = model_train(imgs)  # 得到train_batch个图片的预测输出
        loss = criterion_class(outputs, labels)  # 计算loss

        if args.freeze_layers:
            loss = loss.requires_grad_()

        optimizer.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()

        batch_time.update(time.time() - end)  # 计算每个train_batch所需要的时间
        end = time.time()

        losses.update(loss.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training', loss)
            sys.exit(1)

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch + 1, batch_idx + 1, len(train_loader),
                                                                  batch_time=batch_time, data_time=data_time,
                                                                  loss=losses))


@torch.no_grad()
def evaluate_someone_epoch(epoch, model_train, model, eval_loader, use_gpu):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    model_train.eval()
    if use_gpu:
        top5_num = torch.ones(1).cuda()
        top1_accu_num = torch.zeros(1).cuda()  # 累计预测正确的样本数
        top5_accu_num = torch.zeros(1).cuda()  # 累计预测正确的样本数
    else:
        top5_num = torch.ones(1)
        top1_accu_num = torch.zeros(1)  # 累计预测正确的样本数
        top5_accu_num = torch.zeros(1)  # 累计预测正确的样本数

    sample_num = 0  # 计算总数据量
    end = time.time()
    for batch_idx, (imgs, labels) in enumerate(eval_loader):
        if use_gpu:
            imgs, labels = imgs.cuda(), labels.cuda()
        data_time.update(time.time() - end)  # 更新时间
        sample_num += imgs.shape[0]  # 每迭代一次，imgs.shape[0]就val_batch
        pred = model_train(imgs)
        # 计算top1预测正确的数
        pred_classes = torch.max(pred, dim=1)[1]
        top1_accu_num += torch.eq(pred_classes, labels).sum()

        # 计算top5预测正确的数
        sort_, indices = torch.sort(pred, descending=True)
        for i in range(len(pred)):
            if labels[i] in indices[i][0:5]:
                top5_accu_num += top5_num

        batch_time.update(time.time() - end)  # 计算一个val_batch所需的时间
        end = time.time()
        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(epoch + 1, batch_idx + 1, len(eval_loader),
                                                                            batch_time=batch_time, data_time=data_time))
    print("epoch{} -----> top1: {}\ttop5: {}".format(epoch, top1_accu_num.item() / sample_num,
                                                     top5_accu_num.item() / sample_num))
    return top1_accu_num.item() / sample_num, top5_accu_num.item() / sample_num


def main():
    start_epoch = 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_gpu = torch.cuda.is_available()  # 判断是否有GPU

    if args.use_cpu: use_gpu = False

    pin_memory = True if use_gpu else False  # 节省显存空间


    if use_gpu:
        print("Currently using GPU {}".format(args.device))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    dataset = ImageNet100(root=args.root)

    # 将图片数据进行增强并转化为tensor
    transform_train = T.Compose([
        T.Undistorted2Resize(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = T.Compose([
        T.Undistorted2Resize(args.height, args.width),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    nw = min([os.cpu_count(), args.train_batch if args.train_batch > 1 else 0,
              args.num_workers])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 定义模型
    if args.model == "init":
        print("training convNeXt in ImageNet100!\n")
        model = convnext_base(num_classes=dataset.num_classes)
    elif args.model == "ca":
        print("training convNeXtCa in ImageNet100!\n")
        model = convnextca_base(num_classes=dataset.num_classes)
        args.save_dir = args.save_dir + "_ca"
    elif args.model == "sa":
        print("training convNeXtSa in ImageNet100!\n")
        model = convnextsa_base(num_classes=dataset.num_classes)
        args.save_dir = args.save_dir + "_sa"
    elif args.model == "casa":
        print("training convNeXtCaSa in ImageNet100!\n")
        model = convnextcasa_base(num_classes=dataset.num_classes)
        args.save_dir = args.save_dir + "_casa"
    else:
        raise ValueError

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))

    # 是否从某一个模型文件开始训练,即读档训练
    if args.resume:
        print("Loading checkpoint from {}".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    # 载入预训练模型
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        print("Loading weights from {}".format(args.weights))
        weight_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weight_dict.keys()):
            if "head" in k:
                del weight_dict[k]
            elif "norm.weight" == k:
                del weight_dict[k]
            elif "norm.bias" == k:
                del weight_dict[k]
        weight_dict.update(weight_dict)
        print(model.load_state_dict(weight_dict, strict=False))

    model_train = model.train()
    if use_gpu:
        # model = nn.DataParallel(model)
        model_train = torch.nn.DataParallel(model)
        model_train = model_train.cuda()
        # model.train().cuda()

    start_time = time.time()
    train_time = 0
    # best_acc = -np.inf
    best_top1 = -np.inf
    best_top5 = -np.inf
    is_best = True
    best_epoch = 0

    if True:
        lr = args.lr_one
        train_batch_size = args.train_batch
        val_batch_size = args.val_batch
        Init_epoch = start_epoch
        freeze_epoch = args.freeze_epoch

        criterion_class = nn.CrossEntropyLoss()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(model_train.parameters(), lr=lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        if args.freeze_layers:
            print("=====> Prepare for {} freeze trunk training!".format(freeze_epoch - Init_epoch))
            model.freeze_backbone()
            # if use_gpu:
            #     model.module.freeze_backbone()
            # else:
            #     model.freeze_backbone()
        else:
            print("=====> start training")
            train_batch_size = train_batch_size // 2
            val_batch_size = val_batch_size // 2

        # 每次训练时读取train_batch个数据
        train_loader = DataLoader(
            ImageNet100Dataset(dataset.train, transform=transform_train),
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin_memory,
            drop_last=True,
        )

        # 每次验证时读取val_batch个数据
        eval_loader = DataLoader(
            ImageNet100Dataset(dataset.val, transform=transform_val),
            batch_size=val_batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=True,
        )

        if Init_epoch < freeze_epoch:
            for epoch in range(Init_epoch, freeze_epoch):
                start_train_time = time.time()
                train_one_eopch(epoch, model_train, model, criterion_class, optimizer, train_loader, use_gpu)
                scheduler.step()
                train_time += round(time.time() - start_train_time)
                if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                        epoch + 1) == args.max_epoch:
                    print("==> starting {} with evaluate".format(epoch))
                    # rank1 = val(model, valloader, use_gpu)
                    top1, top5 = evaluate_someone_epoch(epoch, model_train, model, eval_loader, use_gpu)

                    if best_top1 < top1:
                        best_top1 = top1
                        best_epoch = epoch + 1
                        is_best = True

                    if best_top5 < top5:
                        best_top5 = top5

                    # if use_gpu:
                    #     state_dict = model.module.state_dict()
                    #     # model_save = model.module
                    # else:
                    #     # model_save = model
                    #     state_dict = model.state_dict()
                    state_dict = model.state_dict()
                    save_checkpoint({
                        'state_dict': state_dict,
                        'top1': top1,
                        'top5': top5,
                        'epoch': epoch,
                    }, is_best, osp.join(args.save_dir, 'best_checkpoint_ep' + str(epoch + 1) + '.pth'))

                print("==> Best top1 {:.1%}\tBest top5 {:.1%}\tachieved at epoch {}".format(best_top1, best_top5,
                                                                                            best_epoch))
        else:
            print("It has been trained since the {}, "
                  "which is greater than the number of frozen training and skip the freezing training".format(
                start_epoch))
            args.freeze_epoch = Init_epoch

    if True:
        lr = args.lr_two
        train_batch_size = args.train_batch
        val_batch_size = args.val_batch
        max_epoch = args.max_epoch
        freeze_epoch = args.freeze_epoch

        criterion_class = nn.CrossEntropyLoss()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(model_train.parameters(), lr=lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        # 每次训练时读取train_batch个数据
        train_loader = DataLoader(
            ImageNet100Dataset(dataset.train, transform=transform_train),
            batch_size=train_batch_size // 2,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin_memory,
            drop_last=True,
        )

        # 每次验证时读取val_batch个数据
        eval_loader = DataLoader(
            ImageNet100Dataset(dataset.val, transform=transform_val),
            batch_size=val_batch_size // 2,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=True,
        )

        if args.freeze_layers:
            print("=====> Prepare for {} unfreeze trunk training!".format(max_epoch - freeze_epoch))
            model.freeze_backbone()
            # if use_gpu:
            #     model.module.freeze_backbone()
            # else:
            #     model.freeze_backbone()
        else:
            print("=====> continue training")

        for epoch in range(freeze_epoch, max_epoch):
            start_train_time = time.time()
            train_one_eopch(epoch, model_train, model, criterion_class, optimizer, train_loader, use_gpu)
            scheduler.step()
            train_time += round(time.time() - start_train_time)
            if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                    epoch + 1) == args.max_epoch:
                print("==> starting {} with evaluate".format(epoch))
                # rank1 = val(model, valloader, use_gpu)
                top1, top5 = evaluate_someone_epoch(epoch, model_train, model, eval_loader, use_gpu)

                if best_top1 < top1:
                    best_top1 = top1
                    best_epoch = epoch + 1
                    is_best = True

                if best_top5 < top5:
                    best_top5 = top5

                # if use_gpu:
                #     state_dict = model.module.state_dict()
                #     # model_save = model.module
                # else:
                #     # model_save = model
                #     state_dict = model.state_dict()
                state_dict = model.state_dict()
                save_checkpoint({
                    'state_dict': state_dict,
                    'top1': top1,
                    'top5': top5,
                    'epoch': epoch,
                }, is_best, osp.join(args.save_dir, 'best_checkpoint_ep' + str(epoch + 1) + '.pth'))

            print("==> Best top1 {:.1%}\tBest top5 {:.1%}\tachieved at epoch {}".format(best_top1, best_top5,
                                                                                        best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))

    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if __name__ == '__main__':
    main()
