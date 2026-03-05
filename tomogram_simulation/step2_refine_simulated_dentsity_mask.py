import topaz.mrc as mrc
import os
import numpy as np
from skimage.morphology import erosion, ball, skeletonize
import argparse


name = 'merge data'
help = 'merge the proteins to membranes'

parser = argparse.ArgumentParser(description='merge the proteins to membranes')
parser.add_argument('-m', '--membranes',  help='directory to segmented membranes', dest='membranes')
parser.add_argument('-p', '--proteins', help='snr of applied noise', dest='proteins')
parser.add_argument('-o', '--output', help='output directory of noisy tilt series', dest='output')


def load_mrc(path):
    with open(path, 'rb') as f:
        content = f.read()
    tomo = mrc.parse(content)
    img = np.array(tomo[0])
    img = img.astype(np.float32)
    return img


def write_mrc(x, path):
    with open(path, 'wb') as f:
        mrc.write(f, x)


if __name__ == '__main__':
    args = parser.parse_args()
    den_path_membranes = args.membranes
    den_path_particles = args.proteins
    out_path = args.output
    img_membranes = load_mrc(den_path_membranes)
    img_membranes = np.array(img_membranes)
    img_membranes = (img_membranes - img_membranes.min()) / (img_membranes.max() - img_membranes.min())
    img_particles = load_mrc(den_path_particles)
    img_particles = np.array(img_particles)
    img_particles = (img_particles - img_particles.min()) / (img_particles.max() - img_particles.min())

    # 构建与立方体相切的球形mask（基于img_particles形状）
    nz, ny, nx = img_particles.shape
    cz, cy, cx = ( (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0 )
    r = min(cz, cy, cx)
    zz, yy, xx = np.ogrid[0:nz, 0:ny, 0:nx]
    mask_sphere = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)
    mask_sphere = mask_sphere.astype(np.float32)

    # 保存中间mask
    out_root, out_name = os.path.split(out_path)
    name_no_ext, ext = os.path.splitext(out_name)
    mask_path = os.path.join(out_root, f"{name_no_ext}_mask_sphere.mrc")
    write_mrc(mask_sphere, mask_path)

    # 用非球形区域（mask==0）的img_particles像素均值填充img_membranes的空洞
    mean_non_sphere = np.float32(img_particles[mask_sphere == 0].mean())
    img_membranes[img_membranes == 0] = mean_non_sphere

    write_mrc(img_membranes, out_path)