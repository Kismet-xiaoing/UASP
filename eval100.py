import os
import argparse
from pathlib import Path
import warnings
from collections import OrderedDict
from thop import profile

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from models.resnet_cifar10 import resnet_56,resnet_110
import numpy as np
import utils
import time 
from imbalance_data import cifar10Imbanlance,cifar100Imbanlance,dataset_lt_data

import sys
sys.path.insert(0,'..')

from models.resnet_cifar10 import *
from sklearn.metrics import precision_score, recall_score, f1_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context 

name_base=''

def load_resnet_model(model, oristate_dict, layer):
    cfg = {
        56: [9, 9, 9],
        110: [18, 18, 18],
    }

    state_dict = model.state_dict()

    current_cfg = cfg[layer]
    last_select_index = None

    all_conv_weight = []

    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"

    cnt=1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):

                cnt+=1
                cov_id=cnt

                conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                all_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight =state_dict[name_base+conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    ci = np.load(prefix + str(cov_id) + subfix)
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base+conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[name_base+conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if 'shortcut' in name:
                continue
            if conv_name not in all_conv_weight:
                state_dict[name_base+conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base+name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base+name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_vgg_model(model, oristate_dict):
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    cnt=0
    prefix = args.ci_dir+'/ci_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')

        if isinstance(module, nn.Conv2d):

            cnt+=1
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name_base+name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num:

                cov_id = cnt
                print('loading ci from: ' + prefix + str(cov_id) + subfix)
                ci = np.load(prefix + str(cov_id) + subfix)
                select_index = np.argsort(ci)[orifilter_num-currentfilter_num:]  # preserved filter id
                select_index.sort()

                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base+name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                       state_dict[name_base+name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base+name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name_base+name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def test(val_loader, model, criterion):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    # 添加三个新的Meter用于记录指标
    prec_meter = utils.AverageMeter('Precision', ':6.3f')
    recall_meter = utils.AverageMeter('Recall', ':6.3f')
    f1_meter = utils.AverageMeter('F1', ':6.3f')

    # switch to evaluation mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(logits, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        # 计算全局指标（使用macro平均）
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        # 更新Meter（如果需要逐batch计算可移动至循环内）
        prec_meter.update(precision, len(all_targets))
        recall_meter.update(recall, len(all_targets))
        f1_meter.update(f1, len(all_targets))
        output = ('Results: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.5f} Precision {prec:.3f} Recall {recall:.3f} F1 {f1:.3f}'
                  .format(top1=top1, top5=top5, loss=losses,prec=precision, recall=recall, f1=f1))
        print(output)
    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='E:\datasets\cifar100')
    parser.add_argument('--norm', type=str, default='L2')#Linf L2
    parser.add_argument('--epsilon', type=float, default=8./255.)
    # parser.add_argument('--model', type=str, default='./result/resnet_56/100/model_best.pth.tar', 
    #                     help='./result/resnet_56/1/model_best.pth.tar, ./pretrained_models/resnet_56.pt')
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results_l2')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)
    parser.add_argument('--LT', action='store_true', default=True, help='False')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument(
    '--ci_dir',
    type=str,
    default='./100CI_resnet_100',
    help='ci path')
    parser.add_argument(
    '--sparsity',
    type=str,
    default='[0.]+[0.22]*2+[0.35]*18+[0.45]*36',
    help='sparsity of each conv layer, [0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9,[0.]+[0.15]*2+[0.4]*27')
    parser.add_argument(
    '--arch',
    type=str,
    default='resnet_110',
    choices=('vgg_16_bn','resnet_56','resnet_110','resnet_50'),
    help='architecture to calculate feature maps')
    parser.add_argument(
    '--result_dir',
    type=str,
    default='./result/resnet_110/100_0428random',
    help='results path for saving models and loggers')


    args = parser.parse_args()


    # load model
    # model = ResNet50()
    # ckpt = torch.load(args.model)
    # model.load_state_dict(ckpt)

    if args.sparsity:
        import re
        cprate_str = args.sparsity
        cprate_str_list = cprate_str.split('+')
        pat_cprate = re.compile(r'\d+\.\d*')
        pat_num = re.compile(r'\*\d+')
        cprate = []
        for x in cprate_str_list:
            num = 1
            find_num = re.findall(pat_num, x)
            if find_num:
                assert len(find_num) == 1
                num = int(find_num[0].replace('*', ''))
            find_cprate = re.findall(pat_cprate, x)
            assert len(find_cprate) == 1
            cprate += [float(find_cprate[0])] * num

        sparsity = cprate

    model = eval(args.arch)(sparsity=sparsity).cuda()

    if args.resume:
        checkpoint_dir = os.path.join(args.result_dir, 'model_best.pth.tar')
        print('resuming from finetune model')
        ckpt = torch.load(checkpoint_dir, map_location='cuda:0', weights_only=True)
        model.load_state_dict(ckpt['state_dict'])
    else:
        print('resuming from pretrain model')
        model = eval(args.arch)(sparsity=[0.] * 100).cuda()
        ckpt = torch.load('./pretrained/7437res110.pt', map_location='cuda:0', weights_only=True)
        model.load_state_dict(ckpt)

    input_image_size=32
    input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
    flops, params = profile(model, inputs=(input_image,))
    print('Params: %.2f' % (params))
    print('Flops: %.2f' % (flops))

    model.cuda()
    model.eval()

    # load data
    # transform_list = [transforms.ToTensor()]
    # transform_chain = transforms.Compose(transform_list)
    # item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    # test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = utils.CrossEntropyLabelSmooth(10, 0)
    criterion_smooth = criterion_smooth.cuda()

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768)),])    
    if args.LT:
        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=0.02, train=True, transform=transform_test,file_path=args.data_dir)
        print("load cifar100-LT\n")
    else:
        testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
        print("load cifar100\n")
    
    test_loader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)
    testset0 = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader0 = data.DataLoader(testset0, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)
    

    test(test_loader, model, criterion)
    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load attack    
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
        version=args.version)
    
    l = [x for (x, y) in test_loader0]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader0]
    y_test = torch.cat(l, 0)
    
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2
    
    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.batch_size, state_path=args.state_path)

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)
            
            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))
                
