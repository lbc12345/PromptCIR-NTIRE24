import os
import math
import argparse
import torch


import cv2
import numpy as np
from collections import OrderedDict
from model import MyModel


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--save', action='store_true', help='save images')
    parser.add_argument('--data_out_root', type=str)
    parser.add_argument('--data_lq_root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint/best.pt', help='path to save checkpoints')
    parser.add_argument('--ckpt_path2', type=str, default='./checkpoint/best.pt', help='path to save checkpoints')
    args = parser.parse_args()

    return args

def test(model, img):
    _, _, h_old, w_old = img.size()
    padding = 16*2
    h_pad = (h_old // padding + 1) * padding - h_old
    w_pad = (w_old // padding + 1) * padding - w_old
    img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
    img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
    
    img = tile_eval(model, img, tile=384, tile_overlap=96)
    img = img[..., :h_old, :w_old]
    return img

def tile_eval(model,input_,tile=128,tile_overlap=32):
    b, c, h, w = input_.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h, w).type_as(input_)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
    restored = E.div_(W)

    restored = torch.clamp(restored, 0, 1)
    return restored


args = parse_args()


weight = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)

model = MyModel(decoder=True)

model.load_state_dict(weight)
model = model.cuda()

weight = torch.load(args.ckpt_path2, map_location=lambda storage, loc: storage)

model2 = MyModel(decoder=True)

model2.load_state_dict(weight)
model2 = model2.cuda()

test_path = args.data_lq_root

if args.save:
    output_path = args.data_out_root
    if not os.path.exists(output_path):
        os.makedirs(output_path)


model.eval()

with torch.no_grad():

    for img_n in sorted(os.listdir(test_path)):

        lr = cv2.imread(os.path.join(test_path, img_n))
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(lr.transpose((2, 0, 1)))
        img = torch.from_numpy(img).float()
        img /= 255.
        img = img.unsqueeze(0).cuda()
        # img = model(img)
        E0 = test(model, img)
        E1 = test(model2, img)
        L_t = img.transpose(-2, -1)
        E2 = test(model, L_t).transpose(-2, -1)
        E3 = test(model2, L_t).transpose(-2, -1)
        sr = (E0.clamp_(0, 1) + E1.clamp_(0, 1) + E2.clamp_(0, 1) + E3.clamp_(0, 1)) / 4.0
        
        sr = sr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)

        sr = sr * 255.
        sr = np.clip(sr.round(), 0, 255).astype(np.uint8)
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        
        if args.save:
            cv2.imwrite(os.path.join(output_path, img_n.replace('.jpg', '.png')), sr)
