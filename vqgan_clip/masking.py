import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms

import math
import kornia.augmentation as K

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

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


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

replace_grad = ReplaceGrad.apply
clamp_with_grad = torch.clamp #ClampWithGrad.apply

class BoxCropper(object): 
    def __init__(self, w=0.3, h=0.3):
      self.w, self.h = w, h

    def sample(self, source):
        w, h = int(source.width*self.w), int(source.height*self.h)
        w, h = torch.randint(w//2, w+1, []).item(), torch.randint(h//2, h+1, []).item()
        h = w
        x1 = torch.randint(0, source.width - w + 1, []).item()
        y1 = torch.randint(0, source.height - h + 1, []).item()
        x2, y2 = x1 + w, y1 + h
        box = x1, y1, x2, y2
        crop = source.crop(box)
        mask = torch.zeros([source.size[1], source.size[0]])
        mask[y1:y2, x1:x2] = 1.
        return crop, mask

def sample(source, sampler, model, preprocess, n=64000, batch_size=128):
    n_batches = 0- -n // batch_size  # round up
    t_crop = 0

    model.eval()
    with torch.no_grad():
        for step in tqdm(range(n_batches)):
            t_crop = float(step)/float(n_batches)
            crop_cur = (0.4) * (1- t_crop) + (0.1) * t_crop
            sampler.w = crop_cur
            sampler.h = crop_cur

            batch = []
            for _ in range(batch_size):
                crop, mask = sampler.sample(source)
                batch.append((preprocess(crop).unsqueeze(0).to(next(model.parameters()).device), mask))
            crops = torch.cat([img for img, *_ in batch], axis=0)
            embeddings = model.encode_image(crops).cpu().detach()

            for emb, msk in zip(embeddings, [mask for _, mask, *_ in batch]):
                yield emb, msk
    # return samples

def aggregate(samples, labels, model):
    texts = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(texts).cpu()
    masks = []
    for label, text_emb in zip(labels, text_embeddings):
        text_features = text_emb / text_emb.norm(dim=-1, keepdim=True)
        pixel_sum = torch.ones_like(next(samples)[1])
        samples_per_pixel = torch.ones_like(next(samples)[1])
        # dists = [spherical_dist(text_emb.float(), embedding.float()).item()
        #          for embedding, *_ in samples]
        # min_dist, max_dist = min(dists), max(dists)
        for embedding, mask in samples: # dist, (embedding, mask) in zip(dists, samples):
            image_features = embedding / embedding.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp().to(image_features.device)
            logits_per_image = logit_scale * image_features @ text_features.t()
            dist = logits_per_image.float().exp().item()
            # dist = spherical_dist(text_emb.float(), embedding.float()).item()
            pixel_sum += mask * dist
            samples_per_pixel += mask
        img = pixel_sum / samples_per_pixel
        # img *= 4
        # print(img.max())
        # print(img.min(), img.max())
        img = ((img - img.min()
        ) / img.max()) ** 2 # 0.75
        # img /= img.max()
        #img[img <= 0.001] = 0.
        masks.append((img, label))
    return masks


class MakeCutouts(nn.Module):
    def __init__(self, args, cut_size, cutn):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = args.cut_pow # not used with pooling
        
        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            
            if item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True)) # border, reflection, zeros
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))

        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        # self.noise_fac = False

        # Uncomment if you like seeing the list ;)
        # print(augment_list)

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        cutouts = []

        min_size_width = min(sideX, sideY, self.cut_size)
        lower_bound = float(self.cut_size/min_size_width)

        for _ in range(self.cutn):
            size = int(min_size_width*torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound, 1.))
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)

        return clamp_with_grad(batch, 0, 1)


# This is the original version (No pooling)
class MakeCutoutsOrig(nn.Module):
    def __init__(self, args, cut_size, cutn):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = args.cut_pow # not used with pooling
        
        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            
            if item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Hf':
                augment_list.append(K.RandomHorizontalFlip(p=0.5))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='border', keepdim=True)) # border, reflection, zeros
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))

        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        # self.noise_fac = False

        # Uncomment if you like seeing the list ;)
        # print(augment_list)

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        

        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)

        return clamp_with_grad(batch, 0, 1)

def visualise(source, masks):
    source = TF.to_tensor(source)
    for img, label in masks:
        TF.to_pil_image(source * img[None]).save('mask_temp.png')
        display.display(display.Image('mask_temp.png'))

def save(masks):
    source = torch.ones_like(masks[0])
    for img, label in masks:
        return source * img[None]

