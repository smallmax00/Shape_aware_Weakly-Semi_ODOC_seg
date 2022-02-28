import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from utils.losses import BinaryDiceLoss
from utils import ramps
from lib.network import ODOC_cdr_graph
from torch.optim.lr_scheduler import  StepLR
from utils.util import compute_sdf

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='your_own_path', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='your_own_name', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='your_own_name', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=56,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')


# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=28,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')
parser.add_argument('--dropout', type=float,
                    default=0.3, help='dropout rate')
# costs
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if 'UKBB-CDR' in dataset:
        ref_dict = {"1445": 1445, "50": 730, "40": 578, "30": 433, "20": 289, "10": 145, "5": 73}

    else:
        print("Error")
    return ref_dict[str(patiens_num)]





def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    base_lr = args.base_lr

    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = ODOC_cdr_graph(channel=64, k1=500000, k2=70, dropout=args.dropout)
    model = model.cuda()
    model = nn.DataParallel(model)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.999)
    dice_loss = BinaryDiceLoss()
    mseloss = nn.MSELoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, (sampled_batch, path) in enumerate(trainloader):
            # input, PM, vcdr, B-ROI
            volume_batch, label_batch, label_cdr, label_contour = sampled_batch['image'], sampled_batch['label'], sampled_batch['cdr'], sampled_batch['con']


            volume_batch, label_batch, label_cdr, label_contour = volume_batch.float().cuda(), label_batch.float().cuda(), label_cdr.float().cuda(), label_contour.float().cuda()


            outputs_region, outputs_sdm, outputs_est_boundary, \
            outputs_est_region_sdm, outputs_est_boundary_sdm, \
            outputs_est_cdr_region = model(volume_batch)


            outputs_OC_r = outputs_region[:, 0, ...].unsqueeze(1)
            outputs_OD_r = outputs_region[:, 1, ...].unsqueeze(1)


            outputs_OC_s = outputs_sdm[:, 0, ...].unsqueeze(1)
            outputs_OD_s = outputs_sdm[:, 1, ...].unsqueeze(1)

            outputs_est_OC_b = outputs_est_boundary[:, 0, ...].unsqueeze(1)
            outputs_est_OD_b = outputs_est_boundary[:, 1, ...].unsqueeze(1)

            outputs_est_OC_b_sdm = outputs_est_boundary_sdm[:, 0, ...].unsqueeze(1)
            outputs_est_OD_b_sdm = outputs_est_boundary_sdm[:, 1, ...].unsqueeze(1)

            outputs_est_OC_r_sdm = outputs_est_region_sdm[:, 0, ...].unsqueeze(1)
            outputs_est_OD_r_sdm = outputs_est_region_sdm[:, 1, ...].unsqueeze(1)


            gt_OC_r = label_batch[:, 0, ...].unsqueeze(1)
            gt_OD_r = label_batch[:, 1, ...].unsqueeze(1)

            gt_OC_b = label_contour[:, 0, ...].unsqueeze(1)
            gt_OD_b = label_contour[:, 1, ...].unsqueeze(1)

            # geenrate mSDF GT
            with torch.no_grad():
                gt_dis_oc = compute_sdf(label_batch[:, 0, ...].cpu().numpy())
                gt_dis_od = compute_sdf(label_batch[:, 1, ...].cpu().numpy())
                gt_dis_oc[gt_dis_oc > 0] = 1
                gt_dis_od[gt_dis_od > 0] = 1
                label_sdm_oc = torch.from_numpy(gt_dis_oc).float().cuda()
                label_sdm_od = torch.from_numpy(gt_dis_od).float().cuda()
            gt_OC_s = label_sdm_oc.unsqueeze(1)
            gt_OD_s = label_sdm_od.unsqueeze(1)

            """
            supervised loss
            """

            loss_dice_OC_r = dice_loss(
                outputs_OC_r[:args.labeled_bs], gt_OC_r[:args.labeled_bs])
            loss_dice_OD_r = dice_loss(
                outputs_OD_r[:args.labeled_bs], gt_OD_r[:args.labeled_bs])
            loss_dice_OC_b = dice_loss(
                outputs_est_OC_b[:args.labeled_bs], gt_OC_b[:args.labeled_bs]) + dice_loss(outputs_est_OC_b_sdm[:args.labeled_bs], gt_OC_b[:args.labeled_bs])

            loss_dice_OD_b = dice_loss(
                outputs_est_OD_b[:args.labeled_bs], gt_OD_b[:args.labeled_bs]) + dice_loss(outputs_est_OD_b_sdm[:args.labeled_bs], gt_OD_b[:args.labeled_bs])
            loss_mse_OC_s = mseloss(
                outputs_OC_s[:args.labeled_bs], gt_OC_s[:args.labeled_bs])
            loss_mse_OD_s = mseloss(
                outputs_OD_s[:args.labeled_bs], gt_OD_s[:args.labeled_bs])

            """
            weakly supervised loss
            """

            weakly_sup_loss_cdr1 = mseloss(outputs_est_cdr_region, label_cdr)

            cdr_weight = get_current_consistency_weight(iter_num // 500)

            supervised_loss = (loss_dice_OC_r + loss_dice_OD_r + loss_mse_OC_s + loss_mse_OD_s
                               + loss_dice_OC_b /2 + loss_dice_OD_b/2) / 6 +\
                              cdr_weight * weakly_sup_loss_cdr1 /2

            """
            unsupervised loss--boundary
            """

            consistency_weight = get_current_consistency_weight(iter_num//500)
            consistency_OC_b = dice_loss(outputs_est_OC_b[args.labeled_bs:], outputs_est_OC_b_sdm[args.labeled_bs:])
            consistency_OD_b = dice_loss(outputs_est_OD_b[args.labeled_bs:], outputs_est_OD_b_sdm[args.labeled_bs:])
            consistency_loss_b = (consistency_OC_b + consistency_OD_b) / 2


            """
            unsupervised loss--region
            """

            consistency_loss_r = (dice_loss(outputs_est_OC_r_sdm[args.labeled_bs:], outputs_OC_r[args.labeled_bs:]) +
                                  dice_loss(outputs_est_OD_r_sdm[args.labeled_bs:], outputs_OD_r[args.labeled_bs:])) / 2

            """
            total loss
            """
            consistency_loss = (consistency_loss_b + consistency_loss_r) / 2
            loss = supervised_loss + consistency_weight * consistency_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()


            iter_num = iter_num + 1
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_cdr', weakly_sup_loss_cdr1, iter_num)
            writer.add_scalar('info/loss_supervised_region', (loss_dice_OC_r + loss_dice_OD_r) / 2, iter_num)
            writer.add_scalar('info/loss_supervised_sdm', (loss_mse_OC_s + loss_mse_OD_s) / 2, iter_num)

            writer.add_scalar('info/consistency_loss_r',
                              consistency_loss_r, iter_num)
            writer.add_scalar('info/consistency_loss_b',
                              consistency_loss_b, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_supervised: %f, loss_consistency: %f' %
                (iter_num, loss.item(), supervised_loss.item(), consistency_loss.item()))


            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "your_own_path".format(
        args.exp, args.labeled_num, args.model, args.batch_size, args.dropout)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
