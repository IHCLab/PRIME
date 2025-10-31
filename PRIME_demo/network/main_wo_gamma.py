from __future__ import print_function

import os
from time import time
import numpy as np
import torch
import torch.optim
from torchmetrics.image import TotalVariation

from scipy.io import loadmat, savemat
from os.path import join

from Model_wo_gamma import Model
from common_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

train_dev = torch.device("cpu")
save_path = "./network/result/"


def Loss(Zh, Zm, Zh_model, Zm_model):
    lambda1 = 0.1
    alpha = 0.0001
    b, c, h, w = Zh_model.shape
    mse = torch.nn.MSELoss().type(dtype)
    first_term = mse(Zh, Zh_model)
    second_term= mse(Zm, Zm_model)
    tv = TotalVariation().to(train_dev)
    spe_tv = spectral_TV(Zh_model) / (c * h * w)
    spa_tv = tv(Zh_model) / (c * h * w)
    third_term = spa_tv+ alpha * spe_tv 
    loss = first_term + second_term + lambda1 * third_term
    return loss

## Define closure and optimize
def closure():
    global i, net_input
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    Zh_model = net(net_input)  
    Zh_model=Zh_model.to(train_dev)
    Zm_model = down_sample(Zh_model, D) 

    errG = Loss(Zh, Zm, Zh_model, Zm_model)
    errG.backward()

    train_loss_G.append(errG.item())

    all_dict = {
        "Zh_model": Zh_model.detach().squeeze(0).cpu().permute(1, 2, 0).numpy(),
        "Zm_model": Zm_model.detach().squeeze(0).cpu().permute(1, 2, 0).numpy(),
    }
    savemat(save_path + "DL_result.mat", all_dict)

    torch.save(
        {"net": net.state_dict()},
        join(save_path, "model_epoch.pt"),
    )
    i += 1

    return errG

data_Zh = loadmat("./network/Zh3D.mat")
data_Zm = loadmat("./network/Zm3D.mat")
data_D = loadmat("./network/D.mat")

Zh = data_Zh["Zh3D"].astype(np.float32)
Zm = data_Zm["Zm3D"].astype(np.float32)
D = data_D["D"].astype(np.float32)

first_run = (
    True
    if not os.path.exists("./network/result/model_epoch.pt")
    else False
)

## Set up parameters and net
input_band = Zm.shape[2] 
output_band = Zh.shape[2] 

OPT_OVER = "net"
LR = 0.005
OPTIMIZER = "adam"

# =========================net===============================
net = Model(in_channel = input_band, out_channel = output_band).to(train_dev)
# ===========================================================

t_load1 = time()
if not first_run and os.path.exists(join(save_path, "model_epoch.pt")):
    net.load_state_dict(torch.load(join(save_path, "model_epoch.pt"))["net"])
    num_iter = 30
else:
    num_iter = 100
t_load2 = time()

Zh = (
    torch.from_numpy(Zh)
    .permute(2, 0, 1)
    .type(torch.FloatTensor)
    .unsqueeze(0)
    .to(train_dev)
)
Zm = (  
    torch.from_numpy(Zm)
    .permute(2, 0, 1)
    .type(torch.FloatTensor)
    .unsqueeze(0)
    .to(train_dev)
)
D = torch.from_numpy(D).type(torch.FloatTensor).to(train_dev)

net_input = Zm

train_loss_G = []

i = 0
p = get_params(OPT_OVER, net, net_input.detach().clone())

t1 = time()
optimize(OPTIMIZER, p, closure, LR, num_iter)
t2 = time()

with open(join(save_path, "train_time_file"), "a+") as f:
    f.write(str(t2 - t1 + t_load2 - t_load1) + "\n")