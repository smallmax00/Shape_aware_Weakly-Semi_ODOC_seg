import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.dataset import BaseDataSets
from lib.network import ODOC_cdr_graph


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='your_own_path', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='your_own_path', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='your_own_path', help='model_name')
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
# costs
parser.add_argument('--dropout', type=float,
                    default=0.3, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--viz', type=bool,
                    default=False, help='save_pred_masks')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
snapshot_path = "your_own_path".format(
    args.exp, args.labeled_num, args.model, args.batch_size, args.dropout)
saved_model_path = os.path.join(snapshot_path, args.model + '_' + 'best_model_oc.pth')

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


model = ODOC_cdr_graph(channel=64, k1=5000, k2=70, dropout=args.dropout)
model = model.cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(saved_model_path))
model.eval()


db_test = BaseDataSets(base_dir=args.root_path, split="test")

testloader = DataLoader(db_test, batch_size=1, shuffle=False,
                       num_workers=1)


with torch.no_grad():
    for i_batch, (sampled_batch, path) in enumerate(testloader):

        volume_batch, label_batch, label_contour, label_cdr = sampled_batch['image'], sampled_batch['label'], sampled_batch['con'], sampled_batch['cdr']
        volume_batch = volume_batch.cuda()

        pred_region, pred_sdm, _, _, _, _ = model(volume_batch)

        y_pred_OC_r = pred_region[:, 0, ...].cpu().detach().numpy().squeeze(0)
        y_pred_OC_r = (y_pred_OC_r > 0.5).astype(np.uint8)
        y_pred_OD_r = pred_region[:, 1, ...].cpu().detach().numpy().squeeze(0)
        y_pred_OD_r = (y_pred_OD_r > 0.5).astype(np.uint8)

        y_pred_OC_sdm = pred_sdm[:, 0, ...].cpu().detach().numpy().squeeze(0)
        y_pred_OC_sdm[y_pred_OC_sdm > 0] = 1
        y_pred_OD_sdm = pred_sdm[:, 1, ...].cpu().detach().numpy().squeeze(0)
        y_pred_OD_sdm[y_pred_OD_sdm > 0] = 1

        if args.viz:

            plt.imshow(y_pred_OC_r, cmap='gray')
            plt.show()
            plt.imshow(y_pred_OD_r, cmap='gray')
            plt.show()
            plt.imshow(y_pred_OC_sdm, cmap='gray')
            plt.show()
            plt.imshow(y_pred_OD_sdm, cmap='gray')
            plt.show()

