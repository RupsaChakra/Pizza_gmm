# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from collections import OrderedDict
import re
from turtle import clear
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss_GMM
from ssd_gmm import build_ssd_gmm
from utils.test_voc import *
from active_learning_loop import *
import re
import os
from os.path import exists
from pathlib import Path
import shutil
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import math
import random
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from subset_sequential_sampler import SubsetSequentialSampler
# COCO_ROOT_NEW='/common/users/rc1195/gmm/al-mdn/data/realpizza'
writer=SummaryWriter('runs/new')
random.seed(314) # need to change manually
torch.manual_seed(314) # need to change manually


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='VOC dataset root directory path')
# parser.add_argument('--coco_root', default=COCO_ROOT_NEW,
#                     help='COCO dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=False, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--eval_save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--use_cuda', default=True,
                    help='if True use GPU, otherwise use CPU')
parser.add_argument('--id', default=1, type=int,
                    help='the id of the experiment')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.dataset == 'VOC':
    cfg = voc300_active
else:
    cfg = pizza300_active

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.eval_save_folder):
    os.mkdir(args.eval_save_folder)

def get_file_ele(p):
    with open(p,'r') as f:
        obj=f.read()
        li=obj.split('\n')
        li.pop()
        return li



def create_loaders():
    num_train_images = cfg['num_total_images']
    indices = list(range(num_train_images))
    labeled_set=[]
    unlabeled_set=[]
    if args.resume:
        print('why')
        #looks for labeled data info in image_list folder
        try:
            im_names_file=os.listdir('./image_list')[0]
            with open('./image_list/'+im_names_file,'r') as im:
                li=im.read().split('\n')
                li.pop()
            labeled_set=[int(i) for i in li]
            unlabeled_set=[i for i in range(num_train_images) if str(i) not in li]
        except Exception as e:
            #the file path to the tx file containing the indices of the labeled images
            lab_data=get_file_ele('/common/users/rc1195/old_ver_pizza/al-mdn/data/real_pizza_voc/pizza_data/labeled_2.txt')
            train_data=get_file_ele('/common/users/rc1195/old_ver_pizza/al-mdn/data/real_pizza_voc/pizza_data/ImageSets/Main/trainval.txt')
            labeled_set=[int(i) for i in lab_data]
            unlabeled_set=[i for i in range(num_train_images) if i not in labeled_set]
            f = open("labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt", 'w')
            for i in range(len(labeled_set)):
                f.write(str(labeled_set[i]))
                f.write("\n")
            f.close()
            #Move text file to image list
            var="labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt"
            #src directory is the folder containing file with indices of training images
            src='/common/users/rc1195/old_ver_pizza/al-mdn/'+var
            #dst directory is the path to the image list older in your directory
            dst='/common/users/rc1195/old_ver_pizza/al-mdn/image_list'
            try:
                v=dst+'/'+os.listdir('/common/users/rc1195/old_ver_pizza/al-mdn/image_list')[0]
                print(v)
                os.remove(v)
                print('previous file removed')
            except Exception as e:
                print(e)
            shutil.copy(src,dst)
    else:
        lab_data=get_file_ele('/common/users/rc1195/old_ver_pizza/al-mdn/data/real_pizza_voc/pizza_data/labeled_2.txt')
        train_data=get_file_ele('/common/users/rc1195/old_ver_pizza/al-mdn/data/real_pizza_voc/pizza_data/ImageSets/Main/trainval.txt')
        labeled_set=[int(i) for i in lab_data]
        unlabeled_set=[i for i in range(num_train_images) if i not in labeled_set]
        # unlabeled_set = indices[cfg['num_initial_labeled_set']:]
        f = open("labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt", 'w')
        for i in range(len(labeled_set)):
            f.write(str(labeled_set[i]))
            f.write("\n")
        f.close()
        #Move text file to image list
        var="labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt"
        src='/common/users/rc1195/old_ver_pizza/al-mdn/'+var
        dst='/common/users/rc1195/old_ver_pizza/al-mdn/image_list'
        try:
            v=dst+'/'+os.listdir('/common/users/rc1195/old_ver_pizza/al-mdn/image_list')[0]
            # print(v)
            # os.remove(v)
            print('previous file removed')
        except Exception as e:
            print(e)
        shutil.copy(src,dst)


            

    if cfg['name'] == 'VOC':
        supervised_dataset = VOCDetection(root=args.voc_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        unsupervised_dataset = VOCDetection(args.voc_root, [('pizza_data', 'trainval')],
                                            BaseTransform(300, MEANS),
                                            VOCAnnotationTransform())
        
    else:
        supervised_dataset = COCODetection(root=args.coco_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        unsupervised_dataset = COCODetection(args.coco_root,
                                             transform=BaseTransform(300, MEANS))

    supervised_data_loader = data.DataLoader(supervised_dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             sampler=SubsetRandomSampler(labeled_set),
                                             collate_fn=detection_collate,
                                             pin_memory=True)

    unsupervised_data_loader = data.DataLoader(unsupervised_dataset, batch_size=1,
                                               num_workers=args.num_workers,
                                               sampler=SubsetSequentialSampler(unlabeled_set),
                                               collate_fn=detection_collate,
                                               pin_memory=True)
    
    return supervised_dataset, supervised_data_loader, unsupervised_dataset, unsupervised_data_loader, indices, labeled_set, unlabeled_set


def change_loaders(
    supervised_dataset,
    unsupervised_dataset,
    labeled_set,
    unlabeled_set,
):
    supervised_data_loader = data.DataLoader(supervised_dataset, batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             sampler=SubsetRandomSampler(labeled_set),
                                             collate_fn=detection_collate,
                                             pin_memory=True)

    unsupervised_data_loader = data.DataLoader(unsupervised_dataset, batch_size=1,
                                               num_workers=args.num_workers,
                                               sampler=SubsetSequentialSampler(unlabeled_set),
                                               collate_fn=detection_collate,
                                               pin_memory=True)
    return supervised_data_loader, unsupervised_data_loader


def adjust_learning_rate(
    optimizer,
    gamma,
    step,
):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def load_net_optimizer_multi(cfg):
    net = build_ssd_gmm('train', cfg['min_dim'], cfg['num_classes'])

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net = nn.DataParallel(net)
        try:
            net.load_state_dict(torch.load(args.resume))#net.load_state_dict(torch.load(args.resume),strict=False)
        except Exception as e:
            print(e)
            # if re.search('100000v4.pth$',args.resume) or re.search('110000v4.pth$',args.resume) or re.search('120000v4.pth$',args.resume):
            m_state_dict=torch.load(args.resume)
            new_dict=OrderedDict()
            for k,v in m_state_dict.items():
                if 'module.module.' in k:
                    k=k.replace('module.module.','module.')
                new_dict[k]=v
            net.load_state_dict(new_dict)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        net.vgg.load_state_dict(vgg_weights)
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        net.extras.apply(weights_init)
        net.loc_mu_1.apply(weights_init)
        net.conf_mu_1.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    if args.cuda:
        net = nn.DataParallel(net)
        net = net.cuda()
    return net, optimizer

# model = nn.DataParallel(model, device_ids = [2, 0, 1, 3])
# model.to(f'cuda:{model.device_ids[0]}')

def test_save(state_dict,labeled_set,iteration):
    net = build_ssd_gmm('test', cfg['min_dim'], cfg['num_classes'])
    net = nn.DataParallel(net)
    print('loading trained weight at {}...'.format(iteration))
    try:
        net.load_state_dict(state_dict)
    except Exception as e:
        print(e)
        new_state_dict=state_dict
        new_dict=OrderedDict()
        for k,v in new_state_dict.items():
            if 'module.module.' in k:
                k=k.replace('module.module.','module.')
            new_dict[k]=v
        net.load_state_dict(new_dict)
    net.eval()
    test_dataset = VOCDetection(args.voc_root, [('pizza_data', 'test')], BaseTransform(300, MEANS), VOCAnnotationTransform())
    mean_ap = test_net(args.eval_save_folder, net, args.cuda,
                    test_dataset, BaseTransform(300, MEANS),
                    args.top_k, 300, thresh=args.confidence_threshold)
    #save the map value for each cycle for each corresponiding weights from list_iter

    f_path='/common/users/rc1195/old_ver_pizza/al-mdn/'+args.eval_save_folder+'/ssd300_AL_' + cfg['name'] + '_id_' + str(args.id) + '_num_labels_' + str(len(labeled_set)) + '_' + repr(iteration + 1)+'val.txt'
    # Path(f_path).touch()
    with open(f_path,'w') as obj:
        obj.write('mAP value : '+f'{mean_ap}')
        obj.write('/n') 

        
def train(
    labeled_set,
    supervised_data_loader,
    indices,
    cfg,
    criterion,
):
    print(len(labeled_set))
    finish_flag = True
    while finish_flag:
        net, optimizer = load_net_optimizer_multi(cfg)
        net = net.train()
        loc_loss = 0
        conf_loss = 0
        supervised_flag = 1
        step_index = 0

        batch_iterator = iter(supervised_data_loader)
        for iteration in range(args.start_iter, cfg['max_iter']):
            # warm-up
            if iteration < 1000:
                lr = args.lr * ((iteration+1)/1000.0)**4
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(supervised_data_loader)
                images, targets = next(batch_iterator)

            images = images.to(device)
            targets = [ann.to(device) for ann in targets]

            # forward
            t0 = time.time()
            out = net(images)

            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.data
            conf_loss += loss_c.data
            #Iteration vs Loss
            X=iteration
            Y=loss.data
            
            if (float(loss) > 100) or torch.isinf(loss) or torch.isnan(loss):
                # if the net diverges, go back to point 0 and train from scratch
                print('problem here')
                print(float(loss))
                break

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')
                print('loss: %.4f , loss_c: %.4f , loss_l: %.4f , lr : %.4f\n' % (
                      loss.data, loss_c.data, loss_l.data, float(optimizer.param_groups[0]['lr'])))
                writer.add_scalar('AL:Loss/train',Y,X)
            if iteration != 0 and (iteration + 1) % 500 == 0:
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(), 'weights/ssd300_AL_' + cfg['name'] + '_id_' + str(args.id) +
                           '_num_labels_' + str(len(labeled_set)) + '_' + repr(iteration + 1) + '.pth')
                print('**************Model saved*******************')
                test_save(net.state_dict(),labeled_set,iteration)
                print('*******map value save in eval*****************')
            if ((iteration + 1) == cfg['max_iter']):
                finish_flag = False
        else:
            finish_flag = False
    return net

def update_labels(diff):
    if exists('/common/users/rc1195/old_ver_pizza/al-mdn/data/real_pizza_voc/pizza_data/updated_anno'):
        files=os.listdir('/common/users/rc1195/old_ver_pizza/al-mdn/data/real_pizza_voc/pizza_data/updated_anno')
        if len(files)>900:
            src='/common/users/rc1195/old_ver_pizza/al-mdn/data/real_pizza_voc/pizza_data/updated_anno'
            target='./real_pizza_voc/pizza_data/Annotations'
            for file in files:
                shutil.copy2(os.path.join(src,file),target)
            shutil.rmtree(src)
            return False

    return True

def updated_set(old_labeled_set):
    labeled_set=old_labeled_set
    unlabeled_set=[]
    if exists('/common/users/rc1195/old_ver_pizza/al-mdn/data/real_pizza_voc/pizza_data/updated_anno'):
        files=os.listdir('/common/users/rc1195/old_ver_pizza/al-mdn/data/real_pizza_voc/pizza_data/updated_anno')
        train_data=get_file_ele('/common/users/rc1195/old_ver_pizza/al-mdn/data/real_pizza_voc/pizza_data/ImageSets/Main/trainval.txt')
        # labeled_set = indices[:cfg['num_initial_labeled_set']]
        for i in train_data:
            if i+'.xml' in files:
                labeled_set.append(train_data.index(i))
            else:
                unlabeled_set.append(train_data.index(i))
    return labeled_set,unlabeled_set


def main():
    if args.cuda:
        cudnn.benchmark = True
    print(args)
    supervised_dataset, supervised_data_loader, unsupervised_dataset, unsupervised_data_loader, indices, labeled_set, unlabeled_set = create_loaders()
    criterion = MultiBoxLoss_GMM(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    print(len(labeled_set), len(unlabeled_set))
    net = train(labeled_set, supervised_data_loader, indices, cfg, criterion)
    i=0
    if args.resume:
            filename=os.listdir('./image_list')[0]
            curr_train_size=int(filename.split('_')[4])
            i=(curr_train_size-cfg['num_initial_labeled_set'])/1000
    # # active learning loop
    while i!=cfg['num_cycles']:

        if cfg['name'] == 'VOC':
            #select the best weight
            # list_iter = ['12000', '12500', '13000', '13500','14000','14500','15000','15500','16000','16500','17000','17500','18000','18500','19000','19500','20000']
            list_iter=[i for i in range(12000,35500,500)]
            # list_iter=['90000','100000']
            list_weights = []
            for loop in list_iter:
                name = 'weights/ssd300_AL_' + cfg['name'] + '_id_' + str(args.id) + '_num_labels_' + str(len(labeled_set)) + '_' + str(loop) + '.pth'
                list_weights.append(str(name))

            list_mean = []
            for loop in list_weights:
                net = build_ssd_gmm('test', cfg['min_dim'], cfg['num_classes'])
                net = nn.DataParallel(net)
                print('loading trained weight {}...'.format(loop))
                try:
                    net.load_state_dict(torch.load(loop))
                except Exception as e:
                    print(e)
                    new_state_dict=torch.load(loop)
                    new_dict=OrderedDict()
                    for k,v in new_state_dict.items():
                        if 'module.module.' in k:
                            k=k.replace('module.module.','module.')
                        new_dict[k]=v
                    net.load_state_dict(new_dict)
                net.eval()
                test_dataset = VOCDetection(args.voc_root, [('pizza_data', 'test')], BaseTransform(300, MEANS), VOCAnnotationTransform())
                mean_ap = test_net(args.eval_save_folder, net, args.cuda,
                                   test_dataset, BaseTransform(300, MEANS),
                                   args.top_k, 300, thresh=args.confidence_threshold)
                #save the map value for each cycle for each corresponiding weights from list_iter
                try:
                    f_path='/common/users/rc1195/old_ver_pizza/al-mdn/'+args.eval_save_folder+'cycle_num_'+str(i)+'_weight_index_'+str(list_weights.index(loop)+i)+'_val.txt'
                    print(str(list_weights.index(loop)+i))
                    print(i)
                    # Path(f_path).touch()
                    with open(f_path,'w') as obj:
                        obj.write('mAP value : '+f'{mean_ap}')
                        obj.write('/n')
                except Exception as e:
                    print('error saving mAP value')
                    print(e)
                print('1st viz')
                writer.add_scalar('mAP/weights',mean_ap,list_weights.index(loop)+i)
                list_mean.append(float(mean_ap))
            best_weight = list_weights[list_mean.index(max(list_mean))]
            try:
                f_path='/common/users/rc1195/old_ver_pizza/al-mdn/'+args.eval_save_folder+'cycle_num_'+str(i)+'bestmAP_val.txt'
                # Path(f_path).touch()
                with open(f_path,'w') as obj:
                    obj.write('best mAP value : '+str(max(list_mean)))
                    obj.write('/n')
                    obj.write('best weight at : '+str(list_iter[list_mean.index(max(list_mean))]))
            except Exception as e:
                print('error saving mAP value')
                print(e)
            #Visualilzing mAP vs cycle num
            Y=max(list_mean)
            writer.add_scalar('mAP/test',Y,i)
            writer.close()
            # active learning
            net = build_ssd_gmm('train', cfg['min_dim'], cfg['num_classes'])
            net = nn.DataParallel(net)
            print('loading best weight {}...'.format(best_weight))
            try:
                net.load_state_dict(torch.load(best_weight))
            except Exception as e:
                    print(e)
                    new_state_dict=torch.load(loop)
                    new_dict=OrderedDict()
                    for k,v in new_state_dict.items():
                        if 'module.module.' in k:
                            k=k.replace('module.module.','module.')
                        new_dict[k]=v
                    net.load_state_dict(new_dict)

        print('before')
        net.eval()
        print('after')
        batch_iterator = iter(unsupervised_data_loader)
        print('next after')
        old_labeled_set=labeled_set

        labeled_set, unlabeled_set = active_learning_cycle(
            batch_iterator,
            labeled_set,
            unlabeled_set,
            net,
            cfg["num_classes"],
            acquisition_budget=cfg['acquisition_budget'],
            num_total_images=cfg['num_total_images'],
        )
        print('next next after')
        # save the labeled training set list
        print("labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt")
        f = open("labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt", 'w')
        for i in range(len(labeled_set)):
            f.write(str(labeled_set[i]))
            f.write("\n")
        f.close()
        print('text file written and saved')
        #Move text file to image list
        var="labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt"
        src='/common/users/rc1195/old_ver_pizza/al-mdn/'+var
        dst='/common/users/rc1195/old_ver_pizza/al-mdn/image_list'
        try:
            v=dst+'/'+os.listdir('/common/users/rc1195/old_ver_pizza/al-mdn/image_list')[0]
            # print(v)
            os.remove(v)
            print('previous file removed')
        except Exception as e:
            print(e)
        shutil.copy(src,dst)
        diff=[i for i in labeled_set if i not in old_labeled_set]
        while update_labels(diff):
            print("*********Waiting for annotation file****************")
            time.sleep(1200)
            print("*********Waiting for annotation file****************")
        #change labeled_set depending on annotations
        labeled_set,unlabeled_set=updated_set(old_labeled_set)
        # save the labeled training set list
        print("labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt")
        f = open("labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt", 'w')
        for i in range(len(labeled_set)):
            f.write(str(labeled_set[i]))
            f.write("\n")
        f.close()
        print('text file written and saved')
        #Move text file to image list
        var="labeled_training_set_" + cfg['name'] + '_' + str(len(labeled_set)) + '_id_' + str(args.id) + ".txt"
        src='/common/users/rc1195/old_ver_pizza/al-mdn/'+var
        dst='/common/users/rc1195/old_ver_pizza/al-mdn/image_list'
        try:
            v=dst+'/'+os.listdir('/common/users/rc1195/old_ver_pizza/al-mdn/image_list')[0]
            # print(v)
            os.remove(v)
            print('previous file removed')
        except Exception as e:
            print(e)
        shutil.copy(src,dst)
        # change the loaders
        supervised_data_loader, unsupervised_data_loader = change_loaders(supervised_dataset, unsupervised_dataset, labeled_set, unlabeled_set)
        print(len(labeled_set), len(unlabeled_set))

        args.resume = None
        args.start_iter = 0
        net = train(labeled_set, supervised_data_loader, indices, cfg, criterion)
        i+=1
if __name__ == '__main__':
    main()
