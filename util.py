import time
from pathlib import Path

import torch


def embed_background(img):
    backround = torch.ones_like(img)[:, :3, :, :]
    alpha = img[:, 3, :, :]
    return backround * (1 - alpha[:, None, :, :]) + img[:, :3, :, :] * alpha[:, None, :, :]


def embed_color(img: torch.Tensor, color, box_size=70):
    shp = img.shape
    D2 = [shp[2] - box_size, shp[2]]
    D3 = [shp[3] - box_size, shp[3]]
    img = img.clone()
    img[:, :3, D2[0]:D2[1], D3[0]:D3[1]] = color[:, :, None, None]
    if img.shape[1] == 4:
        img[:, -1, D2[0]:D2[1], D3[0]:D3[1]] = 1
    return img


def xyz2tensor(txt, append_normals=False):
    pts = []
    for line in txt.split('\n'):
        line = line.strip()
        line = line.lstrip('v ')
        spt = line.split(' ')
        if 'nan' in line:
            continue
        if len(spt) == 6:
            pts.append(torch.tensor([float(x) for x in spt]))
        if len(spt) == 3:
            t = [float(x) for x in spt]
            if append_normals:
                t += [0.0 for _ in range(3)]
            pts.append(torch.tensor(t))

    rtn = torch.stack(pts, dim=0)
    return rtn


def read_xyz_file(path: Path):
    with open(path, 'r') as file:
        return xyz2tensor(file.read(), append_normals=True)


class RunningAverage:
    def __init__(self):
        self.sum = 0
        self.squre = 0
        self.count = 0

    def add(self, x):
        self.sum += x
        self.squre += x ** 2
        self.increment()

    def reset(self):
        self.sum = 0
        self.count = 0

    def increment(self):
        self.count += 1

    def get_average(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count

    def get_var(self):
        if self.count == 0:
            return 0
        return (float(self.squre) / self.count) - self.get_average() ** 2


def timer_factory():
    class MyTimer(object):
        total_count = 0

        def __init__(self, msg, count=True):
            self.msg = msg
            self.count = count

        def __enter__(self):
            try:
                self.start = time.clock()
            except AttributeError:
                self.start = time.perf_counter()
            print(f'started: {self.msg}')
            return self

        def __exit__(self, typ, value, traceback):
            try:
                self.duration = time.clock() - self.start
            except AttributeError:
                self.duration = time.perf_counter() - self.start
            if self.count:
                MyTimer.total_count += self.duration
            print(f'finished: {self.msg}. duration: {MyTimer.convert_to_time_format(self.duration)}')

        @staticmethod
        def print_total_time():
            print('\n ----- \n')
            print(f'total time: {MyTimer.convert_to_time_format(MyTimer.total_count)}')

        @staticmethod
        def convert_to_time_format(sec):
            sec = round(sec, 2)
            if sec < 60:
                return f'{sec} [sec]'

            minutes = int(sec / 60)
            remaining_seconds = sec - (minutes * 60)
            remaining_seconds = round(remaining_seconds, 2)
            return f'{minutes}:{remaining_seconds} [min:sec]'

    return MyTimer
