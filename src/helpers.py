import torch
import torch.functional as F
import math

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]



def spherical_dist(x, y, noise = False, noise_coeff=0.1):
    x_normed = F.normalize(x, dim=-1)
    y_normed = F.normalize(y, dim=-1)
    if noise:
        with torch.no_grad():
            noise1 = torch.empty(x_normed.shape).normal_(0,0.0422).to(x_normed).detach()*noise_coeff
            noise2 = torch.empty(y_normed.shape).normal_(0,0.0422).to(x_normed).detach()*noise_coeff

            x_normed += noise1
            y_normed += noise2
    x_normed = F.normalize(x_normed, dim=-1)
    y_normed = F.normalize(y_normed, dim=-1)

    return x_normed.sub(y_normed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
    
def bdot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    b = b.expand(B, -1)
    #print(a.shape)
    #print(b.shape)
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

def inner_dist(x,y):
    x_normed = F.normalize(x, dim=-1)
    y_normed = F.normalize(y, dim=-1)
    return bdot(x_normed, y_normed)


def load_vqgan_model(config_path, checkpoint_path):
    global gumbel
    gumbel = False
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)