import torch
from torch import optim
from torch_optimizer import DiffGrad, AdamP, RAdam

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
      
replace_grad = ReplaceGrad.apply

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
      
clamp_with_grad = ClampWithGrad.apply

def get_opt(opt_name, opt_lr):
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
    return opt
