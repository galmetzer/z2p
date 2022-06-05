import argparse
import json
import math
from pathlib import Path
from typing import List

import cv2
import gdown
import numpy as np
import torch
from matplotlib import pyplot as plt

import render_util
import util
from models import PosADANet

ROOT_PATH = Path(__file__).resolve().absolute().parent
torch.manual_seed(10)


def export_results(opts, names: List[str], generated: torch.Tensor):
    for i, name in enumerate(names):
        name = name.replace('.npy', '')
        name = f'{name}.png'

        export_path = opts.export_dir / name
        o_img = generated[i].permute(1, 2, 0)

        o_img = o_img.cpu().numpy() * 255
        cv2.imwrite(str(export_path), o_img)


def load_metadata():
    with open(ROOT_PATH / 'models' / 'default_settings.json', 'r') as file:
        models_meta = json.load(file)
    return models_meta


def load_pretrained_model(model_type: str):
    """
    Load model from memory, or download from drive
    :param model_type: model type as written in default_settings.json
    :return: returns the pretrained model
    """
    models_meta = load_metadata()
    if model_type not in models_meta.keys():
        raise ValueError(f'no model type {model_type}')

    path = ROOT_PATH / models_meta[model_type]['path']
    url = models_meta[model_type]['url']
    num_controls = models_meta[model_type]['len_style']

    if not path.exists():
        gdown.download(url, str(path), quiet=False)
    device = torch.device('cpu')
    model = PosADANet(1, 4, num_controls, padding='zeros', bilinear=True).to(device)
    model.load_state_dict(torch.load(path, map_location=device))

    return model


def load_model_from_checkpoint(opts):
    device = torch.device('cpu')
    models_meta = load_metadata()
    if opts.model_type not in models_meta.keys():
        raise ValueError(f'no model type {opts.model_type}')

    num_controls = models_meta[opts.model_type]['len_style']
    model = PosADANet(1, 4, num_controls, padding=opts.padding, bilinear=not opts.trans_conv).to(device)
    model.load_state_dict(torch.load(opts.checkpoint, map_location=device))
    return model


def generate_controls_vector(opts, n_style):
    # create an empty style vector
    controls = torch.zeros(n_style).float()

    # RGB
    controls[0], controls[1], controls[2] = opts.rgb[2], opts.rgb[1], opts.rgb[0]
    controls[:3] /= 255
    controls[0:3] = controls[0:3].clamp(0, 0.95)

    # Light position
    # delta_r, delta_phi, delta_theta
    controls[3], controls[4], controls[5] = opts.light[0], opts.light[1], opts.light[2]

    # limit phi
    controls[4] = controls[4].clamp(-math.pi / 4, math.pi / 4)
    # limit theta
    controls[5] = controls[5].clamp(0, math.pi / 4)

    if n_style == 8:
        # if in metal roughness mode set metal roughness values as well
        controls[6], controls[7] = opts.metal, opts.roughness
        # limit values between 0-1 i.e. the values the network was trained on
        controls[6] = controls[6].clamp(0, 1)
        controls[7] = controls[7].clamp(0, 1)

    return controls


def main(opts):
    timer = util.timer_factory()
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

    with timer('load pc'):
        pc = util.read_xyz_file(opts.pc)
        pc = render_util.rotate_pc(pc, opts.rx, opts.ry, opts.rz)
        zbuffer = render_util.draw_pc(pc, radius=3, dy=opts.dy, scale=opts.scale)

    if opts.flip_z:
        zbuffer = np.flip(zbuffer, axis=0).copy()

    if opts.show_results:
        plt.imshow(zbuffer)
        plt.show()
    zbuffer: torch.Tensor = torch.from_numpy(zbuffer).float().to(device)

    zbuffer = zbuffer.unsqueeze(-1).permute(2, 0, 1)
    zbuffer: torch.Tensor = zbuffer.float().to(device).unsqueeze(0)

    export_results_flag = opts.export_dir is not None
    if export_results_flag:
        opts.export_dir.mkdir(exist_ok=True, parents=True)
        export_results(opts, [f'zbuffer'], zbuffer.detach())

    if opts.checkpoint:
        # Load model from .pt checkpoint file
        model = load_model_from_checkpoint(opts).to(device)
    else:
        # Download and load published pretrained models
        model = load_pretrained_model(opts.model_type).to(device)
    model.eval()

    controls = generate_controls_vector(opts, model.n_style).to(device)
    controls = controls.unsqueeze(0)

    with torch.no_grad():
        generated = model(zbuffer.float(), controls).clamp(0, 1)
    generated = util.embed_color(generated.detach(), controls[:, :3], box_size=50)

    if opts.show_results:
        plt.imshow(generated[0].permute(1, 2, 0).cpu())
        plt.show()

    if export_results_flag:
        export_results(opts, [f'rendered'], generated.detach())

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--export_dir', type=Path, default=None, required=False,
                        help='path to export directory, if blank dont save')

    parser.add_argument('--pc', type=Path, required=True, help='path to input point cloud to visualize')
    parser.add_argument('--trans_conv', action='store_true',
                        help='use a model with transconv instead of bilinear upsampling')
    parser.add_argument('--rgb', nargs='+', default=[255, 255, 255], type=float, required=False,
                        help='color of the visualized object')
    parser.add_argument('--light', nargs='+', default=[0, 0, 0], type=float, required=False,
                        help='light position input: delta_r delta_phi delta_theta')
    parser.add_argument('--metal', type=float, default=0.5, required=False)
    parser.add_argument('--roughness', type=float, default=0.5, required=False)
    parser.add_argument('--padding', default='zeros', type=str, required=False, help='padding type for the model')
    parser.add_argument('--scale', default=1.0, type=float, required=False,
                        help='pc scale before the 2D z-buffer projection')
    parser.add_argument('--rx', default=-1.9, type=float, required=False,
                        help='rotation on the input pc around the x axis')
    parser.add_argument('--ry', default=0.5, type=float, required=False,
                        help='rotation on the input pc around the y axis')
    parser.add_argument('--rz', default=2.0, type=float, required=False,
                        help='rotation on the input pc around the z axis')
    parser.add_argument('--flip_z', action='store_true', help='flip the z axis')

    parser.add_argument('--dy', default=290, type=int, required=False,
                        help='translation of the input pc in the vertical direction of the image')

    parser.add_argument('--model_type', type=str, required=True,
                        help='model type: regular, metal_roughness,'
                             ' used for setting the number of controls and download of pretrained models')
    parser.add_argument('--checkpoint', type=Path, default=None, required=False,
                        help='path to the .pt model checkpoint,'
                             ' if None a pretrained model will be downloaded from gdrive according to --model_type')

    parser.add_argument('--show_results', action='store_true', help='show results with matplotlib')

    main(parser.parse_args())
