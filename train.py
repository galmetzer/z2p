import argparse
from inspect import getmembers, isfunction
from pathlib import Path

import cv2 as cv
import torch
from torch.utils.data import DataLoader

import data as data
import losses
import util
from models import PosADANet

losses_funcs = {}
for val in getmembers(losses):
    if isfunction(val[1]):
        losses_funcs[val[0]] = val[1]


def log_images(path, msg, img_tensor, style):
    img_tensor = util.embed_color(img_tensor, style[:, :3])
    white_img = util.embed_background(img_tensor)
    for i, img in enumerate(white_img):
        img = img.permute(1, 2, 0)
        img = img.clip(0, 1) * 255
        img = img.detach().cpu().numpy()
        cv.imwrite(f'{str(path)}/{msg}_{i}.png', img)


def get_loss_function(lname):
    return losses_funcs[lname]


def train(opts):
    opts.export_dir.mkdir(exist_ok=True, parents=True)
    train_export_dir: Path = opts.export_dir / 'train'
    train_export_dir.mkdir(exist_ok=True)
    test_export_dir: Path = opts.export_dir / 'test'
    test_export_dir.mkdir(exist_ok=True)
    run_name = opts.export_dir.name

    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    train_set = data.GenericDataset(opts.data, splat_size=opts.splat_size, cache=opts.cache,
                                    keys=opts.controls)
    control_vector_length = train_set.control_length()

    test_set = data.GenericDataset(opts.test_data, splat_size=opts.splat_size, cache=opts.cache,
                                   keys=opts.controls)
    test_elements = opts.batch_size * opts.test_batch_size
    test_set = torch.utils.data.random_split(test_set, [test_elements, len(test_set) - test_elements])[0]

    train_loader = DataLoader(train_set, batch_size=opts.batch_size,
                              shuffle=True, num_workers=opts.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=opts.batch_size,
                             shuffle=True, num_workers=opts.num_workers, pin_memory=True)

    num_samples = len(train_loader)
    model = PosADANet(input_channels=1, output_channels=4, n_style=control_vector_length,
                      padding=opts.padding, bilinear=not opts.trans_conv,
                      nfreq=opts.nfreq, magnitude=opts.freq_magnitude).to(device)

    if opts.checkpoint is not None:
        model.load_state_dict(torch.load(opts.checkpoint))

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    global_step = 0
    avg_loss = util.RunningAverage()
    avg_test_loss = util.RunningAverage()
    for epoch in range(opts.start_epoch, opts.epochs):
        avg_loss.reset()
        model.train()
        for i, (img, zbuffer, color) in enumerate(train_loader):
            optimizer.zero_grad()

            img: torch.Tensor = img.float().to(device)
            zbuffer: torch.Tensor = zbuffer.float().to(device)
            color = color.float().to(device)

            generated = model(zbuffer.float(), color)
            loss = 0
            for weight, lname in zip(opts.l_weight, opts.losses):
                loss += weight * get_loss_function(lname)(generated, img)

            if loss != 0:
                loss.backward()
                optimizer.step()

            if global_step % opts.log_iter == 0:
                expanded_z_buffer = zbuffer.repeat((1, 4, 1, 1))
                expanded_z_buffer[:, -1, :, :] = 1.0
                cat_img = torch.cat([img, generated, expanded_z_buffer.clamp(0, 1)], dim=2)
                log_images(train_export_dir, f'train_imgs{global_step}', cat_img.detach(), color)

            global_step += 1

            avg_loss.add(loss.item())
            print(f'{run_name}; epoch: {epoch}; iter: {i}/{num_samples} loss: {loss}')

        model.eval()
        avg_test_loss.reset()
        for i, (img, zbuffer, color) in enumerate(test_loader):
            with torch.no_grad():
                zbuffer = zbuffer.float().to(device)
                color = color.float().to(device)
                img = img.float().to(device)
                generated = model(zbuffer.float(), color)
                test_loss = 0
                for weight, lname in zip(opts.l_weight, opts.losses):
                    test_loss += weight * get_loss_function(lname)(generated, img)
                avg_test_loss.add(test_loss.item())

                expanded_z_buffer = zbuffer.repeat((1, 4, 1, 1))
                expanded_z_buffer[:, -1, :, :] = 1
                cat_img = torch.cat([img, generated, expanded_z_buffer.clamp(0, 1)], dim=2)
                log_images(test_export_dir, f'pairs_epoch_{epoch}', cat_img.detach(), color)

        print(f'average train loss: {avg_loss.get_average()}')

        torch.save(model.state_dict(), opts.export_dir / f'epoch:{epoch}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--test_data', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--nfreq', type=int, default=20)
    parser.add_argument('--freq_magnitude', type=int, default=10)
    parser.add_argument('--log_iter', type=int, default=1000)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--losses', nargs='+', default=['mse', 'intensity'])
    parser.add_argument('--l_weight', nargs='+', default=[1, 1], type=float)
    parser.add_argument('--tb', action='store_true')
    parser.add_argument('--padding', default='zeros', type=str)
    parser.add_argument('--trans_conv', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--splat_size', type=int)
    parser.add_argument('--controls', nargs='+', default=['colors', 'light_sph_relative'])

    train(parser.parse_args())
