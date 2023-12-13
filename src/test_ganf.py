#%%
import os
import argparse
import torch
from models.GANF import GANF
from load_data import *
import numpy as np
from sklearn.metrics import roc_auc_score
# from data import fetch_dataloaders

ROOT = "/home/jbuckelew/workspace/AML_Project"

parser = argparse.ArgumentParser()
# files
parser.add_argument('--output_dir', type=str, 
                    default='/checkpoint')
# restore
parser.add_argument('--graph', type=str, default='None')
parser.add_argument("--mode", type= str, default="test/")
parser.add_argument("--log", type=str, default="/log/")
parser.add_argument('--model', type=str, default='None')
parser.add_argument('--seed', type=int, default=7, help='Random seed to use.')
# made parameters
parser.add_argument('--n_blocks', type=int, default=6, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--batch_norm', type=bool, default=False)
# training params
parser.add_argument('--batch_size', type=int, default=512)

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


print(args)
import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#%%
print("Loading dataset")

train_loader, val_loader, test_loader = load_data(window_size=3, stride_size=3, batch_size=32, num_entities=10)

#%%
model = GANF(args.n_blocks, 1, args.hidden_size, args.n_hidden, dropout=0.0, batch_norm=args.batch_norm)
model = model.to(device)

params_path = f"{ROOT}/checkpoint/{args.model}/{args.model}_best.pt"
graph_path = f"{ROOT}/checkpoint/{args.model}/graph_best.pt"
model.load_state_dict(torch.load(params_path))
A = torch.load(graph_path).to(device)
model.eval()
#%%
loss_test = []
labels = []
with torch.no_grad():
    for x, label in test_loader:

        x = x.to(device)
        loss = -model.test(x, A.data).cpu().numpy()
        loss_test.append(loss)
        labels.append(label)
loss_test = np.concatenate(loss_test)
labels = np.concatenate(labels)
roc_test = roc_auc_score(labels,loss_test)
metrics = {}
metrics['auc'] = roc_test
output_path = ROOT + args.log + args.mode + args.model + '/'
save_json(metrics, f'{output_path}test_results')
print("The ROC score on Traffic dataset is {}".format(roc_test))
# %%
