import argparse
import math
import random

from urllib.request import urlopen
from tqdm import tqdm
import sys
import os

from omegaconf import OmegaConf

from taming.models import cond_transformer, vqgan

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties
torch.backends.cudnn.benchmark = False

from torch_optimizer import DiffGrad, AdamP, RAdam

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio

from PIL import ImageFile, Image, PngImagePlugin, ImageChops
ImageFile.LOAD_TRUNCATED_IMAGES = True

from subprocess import Popen, PIPE
import re

# Supress warnings
import warnings
warnings.filterwarnings('ignore')

# Check for GPU and reduce the default image size if low VRAM
default_image_size = 512  # >8GB VRAM
if not torch.cuda.is_available():
    default_image_size = 256  # no GPU found
elif get_device_properties(0).total_memory <= 2 ** 33:  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
    default_image_size = 318  # <8GB VRAM

def parse():
    
    vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

    vq_parser.add_argument("-aug",  "--augments", nargs='+', action='append', type=str, choices=['Ji','Sh','Gn','Pe','Ro','Af','Et','Ts','Cr','Er','Re'],
                           help="Enabled augments (latest vut method only)", default=[['Af', 'Pe', 'Ji', 'Er']], dest='augments')
    vq_parser.add_argument("-cd",   "--cuda_device", type=str, help="Cuda device to use", default="cuda:0", dest='cuda_device')
    vq_parser.add_argument("-ckpt", "--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=f'checkpoints/vqgan_imagenet_f16_16384.ckpt',
                           dest='vqgan_checkpoint')
    vq_parser.add_argument("-conf", "--vqgan_config", type=str, help="VQGAN config", default=f'checkpoints/vqgan_imagenet_f16_16384.yaml', dest='vqgan_config')
    vq_parser.add_argument("-cpe",  "--change_prompt_every", type=int, help="Prompt change frequency", default=0, dest='prompt_frequency')
    vq_parser.add_argument("-cutm", "--cut_method", type=str, help="Cut method", choices=['original','updated','nrupdated','updatedpooling','latest'],
                           default='latest', dest='cut_method')
    vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')
    vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest='cutn')
    vq_parser.add_argument("-d",    "--deterministic", action='store_true', help="Enable cudnn.deterministic?", dest='cudnn_determinism')
    vq_parser.add_argument("-i",    "--iterations", type=int, help="Number of iterations", default=500, dest='max_iterations')
    vq_parser.add_argument("-ifps", "--input_video_fps", type=float,
                           help="When creating an interpolated video, use this as the input fps to interpolate from (>0 & <ofps)", default=15,
                           dest='input_video_fps')
    vq_parser.add_argument("-ii",   "--init_image", type=str, help="Initial image", default=None, dest='init_image')
    vq_parser.add_argument("-in",   "--init_noise", type=str, help="Initial noise image (pixels or gradient)", default=None, dest='init_noise')
    vq_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts / target image", default=[], dest='image_prompts')
    vq_parser.add_argument("-iw",   "--init_weight", type=float, help="Initial weight", default=0., dest='init_weight')
    vq_parser.add_argument("-lr",   "--learning_rate", type=float, help="Learning rate", default=0.1, dest='step_size')
    vq_parser.add_argument("-m",    "--clip_model", type=str, help="CLIP model (e.g. ViT-B/32, ViT-B/16)", default='ViT-B/32', dest='clip_model')
    vq_parser.add_argument("-nps",  "--noise_prompt_seeds", nargs="*", type=int, help="Noise prompt seeds", default=[], dest='noise_prompt_seeds')
    vq_parser.add_argument("-npw",  "--noise_prompt_weights", nargs="*", type=float, help="Noise prompt weights", default=[], dest='noise_prompt_weights')
    vq_parser.add_argument("-o",    "--output", type=str, help="Output filename", default="output.png", dest='output')
    vq_parser.add_argument("-ofps", "--output_video_fps", type=float,
                           help="Create an interpolated video (Nvidia GPU only) with this fps (min 10. best set to 30 or 60)", default=0, dest='output_video_fps')
    vq_parser.add_argument("-opt",  "--optimiser", type=str, help="Optimiser", choices=['Adam','AdamW','Adagrad','Adamax','DiffGrad','AdamP','RAdam','RMSprop'],
                           default='Adam', dest='optimiser')
    vq_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default=None, dest='prompts')
    vq_parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height) (default: %(default)s)",
                           default=[default_image_size, default_image_size], dest='size')
    vq_parser.add_argument("-sd",   "--seed", type=int, help="Seed", default=None, dest='seed')
    vq_parser.add_argument("-se",   "--save_every", type=int, help="Save image iterations", default=50, dest='display_freq')
    vq_parser.add_argument("-vid",  "--video", action='store_true', help="Create video frames?", dest='make_video')
    vq_parser.add_argument("-vl",   "--video_length", type=float, help="Video length in seconds (not interpolated)", default=10, dest='video_length')
    vq_parser.add_argument("-vsd",  "--video_style_dir", type=str, help="Directory with video frames to style", default=None, dest='video_style_dir')
    vq_parser.add_argument("-zs",   "--zoom_start", type=int, help="Zoom start iteration", default=0, dest='zoom_start')
    vq_parser.add_argument("-zsc",  "--zoom_scale", type=float, help="Zoom scale %", default=0.99, dest='zoom_scale')
    vq_parser.add_argument("-zse",  "--zoom_save_every", type=int, help="Save zoom image iterations", default=10, dest='zoom_frequency')
    vq_parser.add_argument("-zsx",  "--zoom_shift_x", type=int, help="Zoom shift x (left/right) amount in pixels", default=0, dest='zoom_shift_x')
    vq_parser.add_argument("-zsy",  "--zoom_shift_y", type=int, help="Zoom shift y (up/down) amount in pixels", default=0, dest='zoom_shift_y')
    vq_parser.add_argument("-zvid", "--zoom_video", action='store_true', help="Create zoom video?", dest='make_zoom_video')
    
    
    args = vq_parser.parse_args()
    
    if not args.prompts and not args.image_prompts:
        args.prompts = "A cute, smiling, Nerdy Rodent"

    torch.backends.cudnn.deterministic = args.cudnn_determinism

    # Split text prompts using the pipe character (weights are split later)
    if args.prompts:
        # For stories, there will be many phrases
        story_phrases = [phrase.strip() for phrase in args.prompts.split("^")]
    
        # Make a list of all phrases
        all_phrases = []
        for phrase in story_phrases:
            all_phrases.append(phrase.split("|"))
    
        # First phrase
        args.prompts = all_phrases[0]
    
    # Split target images using the pipe character (weights are split later)
    if args.image_prompts:
        args.image_prompts = args.image_prompts.split("|")
        args.image_prompts = [image.strip() for image in args.image_prompts]

    if args.make_video and args.make_zoom_video:
        print("Warning: Make video and make zoom video are mutually exclusive.")
        args.make_video = False
    
    # Make video steps directory
    if args.make_video or args.make_zoom_video:
        if not os.path.exists('steps'):
            os.mkdir('steps')
    
    return args




if __name__ == 'main':
    
    args = parse()

    
