import numpy as np
from numpy import linalg as la
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='private_linear_regression_syntetic')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--seed_num', type=int, default=100, help='random seed')
# parser.add_argument('--sigma', type=float, default=0, help='variance of noise')
args = parser.parse_args()

print(args)

def bern_proj_add_gauss(D, B, eps, delta, r):
    n = D.shape[0]
    p = 0.5
    p_mat = p * torch.ones(r, n)
    R1 = (torch.bernoulli(p_mat) * 2 - 1).to(device)
    R2 = torch.randn(r, d + 1).to(device)
    sigma = 2 * np.sqrt(d / m) * np.sqrt(2 * r * np.log(1.25 / delta)) * B / eps
    return (torch.matmul(R1, D) + sigma * R2)/ np.sqrt(r), R1 / np.sqrt(r), sigma

def add_gauss(D, B, eps, delta):
    sigma = 2 * np.sqrt(d / m) * np.sqrt(2 * np.log(1.25 / delta)) * B / eps
    return (D.cpu() + torch.randn(D.shape[0], D.shape[1])* sigma).to(device) , sigma

def test_loss(X_test, y_test, beta):
    return ((torch.matmul(X_test, beta) - y_test) ** 2).mean()

delta = 1e-5
B = 1
lam = 1e-4
n_test = 10000
eps_list = [0.1, 0.3, 1.0]
m = 5.0
n_list = [10000, 30000, 100000, 300000, 1000000, 3000000]
super_seed = args.seed
device = "cpu"

if not os.path.exists("private_linear_checkpoint"):
    os.mkdir("private_linear_checkpoint")

for seed in tqdm(range(super_seed * args.seed_num, (super_seed + 1) * args.seed_num)):
    for d in [10]:
        for n in n_list:
            print(f"n: {n}")
            for eps in eps_list:
                checkpoint_name = f"private_linear_checkpoint/multiparty_private_linear_regression_CDF_{d}_{eps}_{n}_{m}_{seed}.pth"
                if os.path.exists(checkpoint_name):
                    continue
                torch.manual_seed(seed)
                X = torch.rand(n + n_test, d) * 2 - 1
                beta = (torch.rand(d) * 2 - 1)  / d
                X, beta = X.to(device), beta.to(device)
                Y = torch.matmul(X, beta)
                X, X_test = X[:n], X[n:]
                Y, Y_test = Y[:n], Y[n:]
                D = torch.cat([X, Y.unsqueeze(1)], dim=1)

                test_losses = []

                sigma_eps_delta = (np.sqrt(2 * np.log(1.25 / delta)) / eps)

                # GMRP
                r_list = [
                         int(1 * np.power(n, 0.5) / sigma_eps_delta),
                         int(3 * np.power(n, 0.5) / sigma_eps_delta),
                         int(10 * np.power(n, 0.5) / sigma_eps_delta)
                         ]
                for r in r_list:
                    D_proj, R_proj, _ = bern_proj_add_gauss(D, B, eps, delta, r)
                    X_proj, Y_proj = D_proj[:, :-1], D_proj[:, -1]
                    H_priv = torch.matmul(X_proj.t(), X_proj) / n + lam * torch.eye(d).to(device)
                    H_inv_priv = torch.inverse(H_priv)
                    B2_priv = torch.matmul(X_proj.t() / n, Y_proj)
                    beta_priv = torch.matmul(H_inv_priv, B2_priv)
                    test_losses.append((test_loss(X_test, Y_test, beta_priv).item(), 
                                        torch.norm(beta_priv - beta).item()))
                    del D_proj, X_proj, R_proj, Y_proj, H_priv, H_inv_priv, B2_priv, beta_priv

                #DGM
                D_add, sigma_add = add_gauss(D, B, eps, delta)
                X_add, Y_add = D_add[:, :-1], D_add[:, -1]
                H_priv = torch.matmul(X_add.t(), X_add) / n - (sigma_add ** 2) * torch.eye(d).to(device) + lam * torch.eye(d).to(device)
                H_inv_priv = torch.inverse(H_priv)
                B2_priv = torch.matmul(X_add.t() / n, Y_add)
                beta_priv = torch.matmul(H_inv_priv, B2_priv)
                test_losses.append((test_loss(X_test, Y_test, beta_priv).item(), 
                                    torch.norm(beta_priv - beta).item()))

                #BDGM
                D_add, sigma_add = add_gauss(D, B, eps, delta)
                X_add, Y_add = D_add[:, :-1], D_add[:, -1]
                H_priv = torch.matmul(X_add.t(), X_add) / n + lam * torch.eye(d).to(device)
                H_inv_priv = torch.inverse(H_priv)
                B2_priv = torch.matmul(X_add.t() / n, Y_add)
                beta_priv = torch.matmul(H_inv_priv, B2_priv)
                test_losses.append((test_loss(X_test, Y_test, beta_priv).item(), 
                                    torch.norm(beta_priv - beta).item()))

                # non-private
                H_inv = torch.inverse(torch.matmul(X.t(), X) / n + lam * torch.eye(d).to(device))
                beta_non_priv = torch.matmul(torch.matmul(H_inv, X.t() / n), Y)
                test_losses.append((test_loss(X_test, Y_test, beta_non_priv).item(), torch.norm(beta - beta_non_priv).item()))
                checkpoint = {}
                checkpoint["test_losses"] = test_losses
                torch.save(checkpoint, checkpoint_name)
                torch.cuda.empty_cache() 