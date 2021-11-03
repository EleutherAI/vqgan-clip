import torch
from torch import nn, optim

from torch_optimizer import DiffGrad, AdamP, RAdam


# Set the optimiser
def get_opt(opt_name, z, opt_lr):
    if opt_name == "Adam":
        opt = optim.Adam([z], lr=opt_lr)    # LR=0.1 (Default)
    elif opt_name == "AdamW":
        opt = optim.AdamW([z], lr=opt_lr)   
    elif opt_name == "Adagrad":
        opt = optim.Adagrad([z], lr=opt_lr) 
    elif opt_name == "Adamax":
        opt = optim.Adamax([z], lr=opt_lr)  
    elif opt_name == "DiffGrad":
        opt = DiffGrad([z], lr=opt_lr, eps=1e-9, weight_decay=1e-9) # NR: Playing for reasons
    elif opt_name == "AdamP":
        opt = AdamP([z], lr=opt_lr)         
    elif opt_name == "RAdam":
        opt = RAdam([z], lr=opt_lr)         
    elif opt_name == "RMSprop":
        opt = optim.RMSprop([z], lr=opt_lr)
    else:
        print("Unknown optimiser. Are choices broken?")
        opt = optim.Adam([z], lr=opt_lr)
    return opt, z
