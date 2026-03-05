import os
import topaz.mrc as mrc
import argparse
import numpy as np
import pyfftw.interfaces.numpy_fft as fft
from utils.ctf import *
from tqdm import tqdm
# from utils import *
from numpy.fft import *

def load_mrc(path):
    """Load MRC file and return data and pixel size information"""
    with open(path, 'rb') as f:
        content = f.read()
    tomo = mrc.parse(content)
    img = np.array(tomo[0])
    img = img.astype(float)
    
    # Extract pixel size information from MRC header
    # The pixel size is stored in cella (xlen, ylen, zlen) divided by dimensions
    header = tomo[1]  # Get the header
    if header.xlen > 0 and header.ylen > 0 and header.zlen > 0:
        # Calculate pixel size from cell dimensions
        pixel_size_x = header.xlen
        pixel_size_y = header.ylen
        pixel_size_z = header.zlen
        pixel_size = (pixel_size_x, pixel_size_y, pixel_size_z)
        mx = header.nx
        my = header.ny
        mz = header.nz
        tomo_size = (mx, my, mz)
    else:
        pixel_size = (1.0, 1.0, 1.0)
        tomo_size = (1, 1, 1)
    return img, pixel_size, tomo_size


def write_mrc(x, path, pixel_size=(1.0, 1.0, 1.0), tomo_size=(1, 1, 1)):
    """Write MRC file with specified pixel size using mrc library"""
    try:
        # Write MRC file with custom header
        with open(path, 'wb') as f:
            mrc.write(f, x, ax=pixel_size[0], ay=pixel_size[1], az=pixel_size[2], mx=tomo_size[0], my=tomo_size[1], mz=tomo_size[2])
        
        #print(f"MRC file saved with pixel size: {pixel_size} Angstroms")
    except Exception as e:
        print(f"Warning: Could not set pixel size, saving without pixel size info: {e}")
        # Fallback to original method
        with open(path, 'wb') as f:
            mrc.write(f, x)


name = 'apply ctf to tilts'
help = 'apply simulated ctf to cryo-et tilts'

parser = argparse.ArgumentParser(description='apply simulated ctf to cryo-et tilts')
parser.add_argument('-t', '--tiltpath', help='directory to tilts', dest='tiltpath')
parser.add_argument('-o', '--output', help='directory to tilts applied ctf', dest='output')
parser.add_argument('-d1', '--defocus1', help='the first bound of defocus value', dest='def1')
parser.add_argument('-d2', '--defocus2', help='the second bound of defocus value', dest='def2')
parser.add_argument('-an', '--angleast', help='shift angle on x axis', dest='angast')
parser.add_argument('-kv', '--kvoltage', help='operation voltage of tem', dest='kv')
parser.add_argument('-ac', '--accum', help='operation voltage of tem', dest='ac')
parser.add_argument('-cs', '--sphereabbr', help='spherical abbrevation', dest='cs')


def ctf_modulation(tilt_path, output, def1=1500, def2=1800, angast=1, kv=300, ac=0.1, cs=2.7):

    p_tilts, voxel_size, tomo_size = load_mrc(tilt_path)
    num_tilts = p_tilts.shape[0]
    ctf_applied_tilts = []
    for i in tqdm(range(num_tilts)):
        s, a = ctf_freq(shape=p_tilts[i].shape, d=angast, full=True)
        c = eval_ctf(s, a, def1, def2, angast, kv=kv, ac=ac, cs=cs, bf=100)   # size 1024
        fft_c = fftshift(c)
        proj_ctf = np.real(ifftn(ifftshift(fft_c * fftshift(fftn(p_tilts[i])))))
        ctf_applied_tilts.append(proj_ctf)
    np_ctf_applied_tilts = np.array(ctf_applied_tilts)

    # print('save the ctf modulated tilt series...')
    # if not os.path.exists('./results_tilt'):
    #     os.makedirs('./results_tilt')
    # out_path = os.path.join('./results_tilt', output)

    if not os.path.exists(os.path.dirname(output)):
        os.makedirs(os.path.dirname(output))
    out_path = output
    write_mrc(np_ctf_applied_tilts, out_path, voxel_size, tomo_size)


if __name__ == '__main__':

    args = parser.parse_args()
    tiltpath = args.tiltpath
    output = args.output
    def1 = int(args.def1)
    def2 = int(args.def2)
    angast = np.float32(args.angast)
    kv = int(args.kv)
    ac = np.float32(args.ac)
    cs = np.float32(args.cs)
    ctf_modulation(tiltpath, output, def1, def2, angast, kv, ac, cs)


# python crSim_apply_ctf_to_tilts.py -t ./example_data/tilts_clean_50_angles_3_interval.mrc -o tilts_applied_ctf.mrc -d1 1000 -d2 1300 -an 1 -kv 300 -ac 0.1 -cs 2.7
# python crSim_apply_ctf_to_tilts.py -t ./output_tilts/tomos/mics_tilt_noiseless_60_interval_3.mrc -o tilts_applied_ctf.mrc -d1 1000 -d2 1300 -an 1 -kv 300 -ac 0.1 -cs 2.7

# python crSim_apply_ctf_to_tilts.py -t /data/ts/CrSimBench/10499_data/tomos/mics_tilt_noiseless_60_interval_3.mrc -o /data/ts/CrSimBench/10499_data/tilts_applied_ctf.mrc -d1 1000 -d2 1300 -an 1 -kv 300 -ac 0.1 -cs 2.7