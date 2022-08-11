import healpy as hp
from astropy.table import Table
from scipy.interpolate.ndgriddata import griddata
import numpy as np
from matplotlib import pyplot as plt
import math
import cupy as cp
import h5py

NSIDE = 2048
angle_r = 2.0 * np.pi / 180.0
grid_x, grid_y = np.mgrid[-angle_r:angle_r:128j, -angle_r:angle_r:128j]
gauss_width = 2.0 * angle_r / 128

KERNEL_SOURCE = r'''
constexpr float M_PI = 3.14159265359;
constexpr float angle_r = 2.0 * M_PI / 180.0;
constexpr float gauss_width = 2.0 * angle_r / 128;

extern "C" {
    __global__ void gaussint_kernel(float* out, float* points_x, float* points_y, float* points_val, int* points_from, int N) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int b = blockDim.z * blockIdx.z + threadIdx.z;
        if (i < 128 && j < 128 && b < N) {
            float x = (2.0 * i / 127.0 - 1.0) * angle_r;
            float y = (2.0 * j / 127.0 - 1.0) * angle_r;
            float s = 0.0, sw = 0.0;
            int to = points_from[b + 1];
            for (int k = points_from[b]; k < to; ++k) {
                float dx = x - points_x[k], dy = y - points_y[k];
                float w = exp(-(dx * dx + dy * dy) / (gauss_width * gauss_width));
                s += points_val[k] * w;
                sw += w;
            }
            out[b * 128 * 128 + 128 * i + j] = s / sw;
        }
    }
}
'''
module = cp.RawModule(code=KERNEL_SOURCE, options=('--std=c++17', '--use_fast_math'))
gaussint_kernel = module.get_function('gaussint_kernel')

def sim(seed=None):
    if seed is not None:
        np.random.seed(seed)
    table = Table.read('COM_PowerSpect_CMB-TT-full_R3.01.txt', format='ascii')
    cl = np.array([0, 0] + [x / l / (l + 1) * 2 * np.pi for l, x in zip(table.columns[0], table.columns[1])])# / 0.1000442E+01**2
    return hp.sphtfunc.synfast(cl, NSIDE, pol=False, fwhm=5*np.pi/180/60)

def extract_patch(dir, sim):
    idxs = hp.query_disc(NSIDE, dir, angle_r * np.sqrt(2.0))
    dx = np.random.randn(3)
    dx -= np.dot(dx, dir) * dir
    dx /= np.sqrt(np.dot(dx, dx))
    dy = np.cross(dir, dx)
    poses = np.array(hp.pix2vec(NSIDE, idxs)).T - dir
    vals = sim[idxs]
    xs = np.dot(poses, dx)
    ys = np.dot(poses, dy)
    return xs, ys, vals
    # grid = griddata((xs, ys), vals, (grid_x, grid_y), method='cubic', fill_value=0)

with h5py.File('dataset_planck.hdf5', 'w') as f:
    patches = f.create_dataset(f'patches', shape=(0, 128, 128), dtype=np.float32, maxshape=(None, 128, 128), chunks=(1, 128, 128))
    coords = f.create_dataset(f'coords', shape=(0, 2), dtype=np.float32, maxshape=(None, 2))
    for it in range(12):
        print(it)
        cmb = sim()
        # cmb = hp.read_map('COM_CMB_IQU-smica_2048_R3.00_full.fits')
        points_x, points_y, points_val, points_from = [], [], [], [0]
        points_coords = []
        for i in range(10000):
            dir = np.random.randn(3)
            dir /= np.sqrt(np.dot(dir, dir))
        # for i in range(64 * 64):
        #     dir = hp.pix2vec(64, (it % 12) * 64 * 64 + i)
            # if -np.arcsin(dir[2]) < 22 * np.pi / 180:
            #     continue
            xs, ys, vals = extract_patch(np.array(dir), cmb)
            points_x.append(cp.array(xs, dtype=cp.float32))
            points_y.append(cp.array(ys, dtype=cp.float32))
            points_val.append(cp.array(vals, dtype=cp.float32))
            points_from.append(points_from[-1] + len(vals))
            points_coords.append(hp.pix2ang(64, (it % 12) * 64 * 64 + i, lonlat=True))
        if len(points_from) == 1:
            continue
        out = cp.zeros((len(points_from) - 1, 128, 128), dtype=cp.float32)
        gaussint_kernel((1, 16, len(points_from) - 1), (128, 8, 1), (out, cp.concatenate(points_x), cp.concatenate(points_y), cp.concatenate(points_val), cp.array(points_from, dtype=cp.int32), len(points_from) - 1))
        patches.resize(patches.shape[0] + out.shape[0], axis=0)
        patches[-out.shape[0]:] = out.get()
        coords.resize(coords.shape[0] + out.shape[0], axis=0)
        coords[-out.shape[0]:] = points_coords
        f.flush()
        # plt.imshow(out[-1].get())
        # plt.show()
