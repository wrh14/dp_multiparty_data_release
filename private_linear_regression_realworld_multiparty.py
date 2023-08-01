import numpy as np
import torch
import math
import argparse
import os
os.environ['KMP_WARNINGS'] = '0'
from numpy import linalg as la

def bern_proj_add_gauss(D, B, eps, delta, r):
    n = D.shape[0]
    p = 0.5
    p_mat = p * torch.ones(r, n)
    R1 = (torch.bernoulli(p_mat) * 2 - 1).to(device)
    R2 = torch.randn(r, d + 1).to(device)
    sigma = 2 * np.sqrt(math.ceil(d/m)) * np.sqrt(2 * r * np.log(1.25 / delta)) * B / eps
    return (torch.matmul(R1, D) + sigma * R2)/ np.sqrt(r), R1 / np.sqrt(r), sigma

def add_gauss(D, B, eps, delta):
    sigma = 2 * np.sqrt(math.ceil(d/m)) * np.sqrt(2 * np.log(1.25 / delta)) * B / eps
    return (D.cpu() + torch.randn(D.shape[0], D.shape[1])* sigma).to(device) , sigma

def test_loss(X_test, y_test, beta):
    return ((torch.matmul(X_test, beta) - y_test) ** 2).mean()

def clipped_test_loss(X_test, y_test, beta):
    return ((torch.clamp(torch.matmul(X_test, beta), min=-1, max=1) - y_test) ** 2).mean()


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./data/UCI/', help='data root')
parser.add_argument('--dataset', type=str, default='wine_quality-white', help='data root')
parser.add_argument('--eps', type=float, default=1, help='privacy budget')
parser.add_argument('--delta', type=float, default=1e-5, help='privacy budget')
parser.add_argument('--lam', type=float, default=1e-5, help='regularization')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--sub_trainset', type=float, default=1, help='ratio of subset')
args = parser.parse_args()

eps = args.eps
delta = args.delta
lam = args.lam
B = 1
device = "cpu"

data_checkpoint = torch.load(args.data_root + "/" + args.dataset + "/processed_data.ptl")
X_train, Y_train, X_test, Y_test = data_checkpoint["X_train"], data_checkpoint["Y_train"], data_checkpoint["X_test"], data_checkpoint["Y_test"]
X_train, Y_train, X_test, Y_test = torch.from_numpy(X_train).float().to(device), torch.from_numpy(Y_train).float().to(device), torch.from_numpy(X_test).float().to(device), torch.from_numpy(Y_test).float().to(device)

np.random.seed(args.seed)
n, d = X_train.size()
randperm = np.random.permutation(n)
train_idx = randperm[:int(n * args.sub_trainset)]
X_train, Y_train = X_train[train_idx], Y_train[train_idx]
D = torch.cat([X_train, Y_train.unsqueeze(1)], dim=1)
n, d = X_train.size()

if d < 20:
    m = 5
else:
    m = 10

test_losses = []
beta_list = []

sigma_eps_delta = (np.sqrt(2 * np.log(1.25 / delta)) / eps)

# non-private
H_inv = torch.inverse(torch.matmul(X_train.t(), X_train) / n + lam * torch.eye(d).to(device))
beta = torch.matmul(torch.matmul(H_inv, X_train.t() / n), Y_train)
test_losses.append((test_loss(X_test, Y_test, beta).item(),
                    clipped_test_loss(X_test, Y_test, beta).item(), 
                    0))
beta_list.append(beta)

r_list = [100, 300, 1000, 3000, 10000]
for r in r_list:
    D_proj, R_proj, _ = bern_proj_add_gauss(D, B, eps, delta, r)
    X_proj, Y_proj = D_proj[:, :-1], D_proj[:, -1]
    H_priv = torch.matmul(X_proj.t(), X_proj) / n + lam * torch.eye(d).to(device)
    H_inv_priv = torch.inverse(H_priv)
    B2_priv = torch.matmul(X_proj.t() / n, Y_proj)
    beta_priv = torch.matmul(H_inv_priv, B2_priv)
    test_losses.append((test_loss(X_test, Y_test, beta_priv).item(), 
                        clipped_test_loss(X_test, Y_test, beta_priv).item(), 
                        torch.norm(beta_priv - beta).item()))
    beta_list.append(beta_priv)
    del D_proj, X_proj, R_proj, Y_proj, H_priv, H_inv_priv, B2_priv, beta_priv

#DGM
D_add, sigma_add = add_gauss(D, B, eps, delta)
X_add, Y_add = D_add[:, :-1], D_add[:, -1]
H_priv = torch.matmul(X_add.t(), X_add) / n - (sigma_add ** 2) * torch.eye(d).to(device) + lam * torch.eye(d).to(device)
H_inv_priv = torch.inverse(H_priv)
B2_priv = torch.matmul(X_add.t() / n, Y_add)
beta_priv = torch.matmul(H_inv_priv, B2_priv)
beta_list.append(beta_priv)
test_losses.append((test_loss(X_test, Y_test, beta_priv).item(), 
                    clipped_test_loss(X_test, Y_test, beta_priv).item(), 
                    torch.norm(beta_priv - beta).item()))

#BDGM
D_add, sigma_add = add_gauss(D, B, eps, delta)
X_add, Y_add = D_add[:, :-1], D_add[:, -1]
H_priv = torch.matmul(X_add.t(), X_add) / n + lam * torch.eye(d).to(device)
H_inv_priv = torch.inverse(H_priv)
B2_priv = torch.matmul(X_add.t() / n, Y_add)
beta_priv = torch.matmul(H_inv_priv, B2_priv)
beta_list.append(beta_priv)
test_losses.append((test_loss(X_test, Y_test, beta_priv).item(), 
                    clipped_test_loss(X_test, Y_test, beta_priv).item(),
                    torch.norm(beta_priv - beta).item()))

checkpoint = {}
checkpoint["test_losses"] = test_losses
checkpoint["beta_list"] = beta_list
checkpoint["r_list"] = r_list
torch.save(checkpoint, f"checkpoint/private_feature_regression/{args.dataset}_{args.eps}_{args.delta}_{args.sub_trainset}_{args.lam}_{m}_{args.seed}")