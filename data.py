from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

RANGES = [[0, 540], [100, 960]]
TARGET = [500, -1]


def resize(img):
    if TARGET[1] == -1:
        r = img.shape[0] / img.shape[1]
        img = cv2.resize(img, (TARGET[0], int(r * TARGET[0])))
    else:
        img = cv2.resize(img, (TARGET[0], TARGET[1]))

    return img


class BinMapper:
    """
    Map binned data to global index
    """

    def __init__(self, counts: [int]):
        self.total_elements = sum(counts)
        self.inverse_map = []
        for i, c in enumerate(counts):
            for j in range(c):
                self.inverse_map.append((i, j))

    def __len__(self):
        return len(self.inverse_map)

    def __getitem__(self, item):
        return self.inverse_map[item]


class Shape:

    @staticmethod
    def find_npys(view_folder: Path):
        """
        Find how many examples are for a certain view
        :param view_folder: path to view folder
        :return: int number of examples, 0 if 'render.png' not found
        """
        if not (view_folder / 'render.png').exists():
            return 0

        counter = 0
        for npy in view_folder.iterdir():
            spt = npy.name.split('.')
            all_numbers = all(['0' <= c <= '9' for c in spt[0]])
            if len(spt) == 2 and all_numbers and spt[1] == 'npy':
                # yield npy
                counter += 1
        return counter

    def __init__(self, folder):
        self.folder = Path(folder)
        self.views = []
        self.counts = []

        # find views
        for f in self.folder.iterdir():
            if f.name.startswith('view_'):
                t_count = Shape.find_npys(f)
                # if there are examples for this view add the it to shapes valid views
                if t_count > 0:
                    self.views.append(f)
                    self.counts.append(t_count)

        # Map the examples per view to a global index
        self.inverter = BinMapper(self.counts)

    def __len__(self):
        # This returns the total number of examples for all views
        return len(self.inverter)

    def __getitem__(self, item):
        # map between global index to a specific example in a specific view
        view_index, item_index = self.inverter[item]
        return self.views[view_index] / f'render.png', self.views[view_index] / f'{item_index}.npy'


class GenericDataset(Dataset):
    def __init__(self, folder: Path, keys=('colors', 'light_sph_relative'),
                 splat_size=3, cache=True):
        self.folder = Path(folder)
        self.splat_size = splat_size
        self.shape_paths = list(self.folder.iterdir())
        self.shapes = []
        self.cache = cache
        # read all shapes with positive amount of views
        for s in self.shape_paths:
            t_shape = Shape(s)
            if len(t_shape) > 0:
                self.shapes.append(t_shape)

        self.inverter = BinMapper([len(x) for x in self.shapes])
        self.keys = keys

    def control_length(self):
        _, _, settings = self[0]
        return settings.shape[0]

    def __len__(self):
        # This returns the total number of examples for all shapes
        return len(self.inverter)

    def __getitem__(self, item):
        # map between global index to a specific example in a specific shape
        shape_index, item_index = self.inverter[item]
        img_path, z_buffer_path = self.shapes[shape_index][item_index]
        rtn = load_files(img_path, z_buffer_path, splat_size=self.splat_size, cache=self.cache)
        if rtn is None:
            return self.__getitem__(0)

        settings_path = img_path.parent / 'settings.npy'
        settings_dict = np.load(settings_path, allow_pickle=True).item()

        settings_vector = []
        for k in self.keys:
            if isinstance(settings_dict[k], float):
                attr = torch.tensor([settings_dict[k]])
            else:
                attr = torch.from_numpy(settings_dict[k])

            if k == 'colors':
                attr = torch.flip(attr, [0])
            settings_vector.append(attr)

        settings_vector = torch.cat(settings_vector)
        img, zbuffer = rtn

        return img, zbuffer, settings_vector


def scatter(u_pix, v_pix, distances, res, radius=5, dr=(0, 0), const=6, scale_const=0.7):
    distances -= const
    img = np.zeros(res)
    for (u, v, d) in zip(u_pix, v_pix, distances):
        v, u = round(res[0] - (v - dr[0])), round(u - dr[1])
        f = np.exp(-d / scale_const)
        if radius == 0:
            img[v, u] = max(img[v, u], f)
        else:
            for t1 in range(-radius, radius):
                for t2 in range(-radius, radius):
                    ty, tx = v - t1, u - t2
                    ty, tx = max(0, ty), max(0, tx)
                    ty, tx = min(res[0] - 1, ty), min(res[1] - 1, tx)
                    img[ty, tx] = max(img[ty, tx], f)

    return img


def parse_pts(ar, radius=3, dr=(0, 0), const=6):
    """
    produce a z_buffer from a numpy array of points in relative coordinates
    :param ar: points array NX3 (** with last point representing resolution)
    :param radius: point radius on screen
    :return: np.array NXM with exponential z values for pixel, 0 is inf/background
    """
    res = ar[-1, :-1]
    res = res[::-1].astype(np.int)
    ar = ar[:-1]
    x_target, y_target = int(res[0] / 2), int(res[1] / 2)
    return scatter(ar[:, 0] / 2, ar[:, 1] / 2, ar[:, 2], (x_target, y_target), radius=radius, dr=dr, const=const)


def load_files(png_path, npy_path, splat_size=5, cache=True, dr=(0, 0)):
    """
    load a png target and render a z_buffer from an npy_path
    :param png_path: path to the png target image
    :param npy_path: path to the npy containg the points
    :param cache: if True used caches Z-buffers else False
    :return: (torch.Tensor, torch.Tensor) target_image (3XNXM), z_buffer (1XNXM)
    """
    if png_path is not None:
        img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        img = img[RANGES[0][0]: RANGES[0][1], RANGES[1][0]:RANGES[1][1], :]
        img = resize(img)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1) / 255

    if cache:
        cache_path = npy_path.parent / npy_path.name.replace('.npy', '_cache.npy')
        if cache_path.exists():
            z_buffer = np.load(cache_path)
        else:
            ptsD = np.load(str(npy_path))
            z_buffer = parse_pts(ptsD, radius=splat_size)
            np.save(cache_path, z_buffer)
    else:
        ptsD = np.load(str(npy_path))
        z_buffer = parse_pts(ptsD, radius=splat_size, dr=dr)

    z_buffer = z_buffer[RANGES[0][0]: RANGES[0][1], RANGES[1][0]:RANGES[1][1]]
    z_buffer = resize(z_buffer)

    z_buffer = torch.from_numpy(z_buffer)
    z_buffer = z_buffer.unsqueeze(0)

    if png_path is not None:
        return img, z_buffer
    else:
        return z_buffer
