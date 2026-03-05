import os
import sys
import time
import random
import numpy as np
import argparse

from polnet.utils import *
from polnet import lio
from polnet import tem
from polnet import poly as pp
from polnet.network import NetSAWLC, NetSAWLCInter, NetHelixFiber, NetHelixFiberB
from polnet.polymer import FiberUnitSDimer, MTUnit, MB_DOMAIN_FIELD_STR
from polnet.stomo import MmerFile, MbFile, SynthTomo, SetTomos, HelixFile, MTFile, ActinFile, MmerMbFile
from polnet.lrandom import EllipGen, SphGen, TorGen, PGenHelixFiberB, PGenHelixFiber, SGenUniform, SGenProp, OccGen
from polnet.membrane import SetMembranes

##### Input parameters

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='3D重建脚本')
    parser.add_argument(
        '--out_dir', 
        type=str, 
        default=os.path.realpath(os.getcwd() + '/data_generated/polnet_test_20250728_emd11999_bin8_1024'),
        help='输出目录路径 (默认: ./data_generated/polnet_test_20250728_emd11999_bin8_1024)'
    )
    
    parser.add_argument(
        '--tomo_type', 
        type=str, 
        default=None
    )
    
    parser.add_argument(
        '--tilt_start',
        type=int,
        default=-90,
        help='倾斜角度起始值 (默认: -90)'
    )
    parser.add_argument(
        '--tilt_end',
        type=int,
        default=91,
        help='倾斜角度结束值 (默认: 91)'
    )
    parser.add_argument(
        '--tilt_step',
        type=int,
        default=1,
        help='倾斜角度步长 (默认: 1)'
    )
    return parser.parse_args()

# 解析命令行参数
args = parse_arguments()

# Common tomogram settings
# 指定all_features_of_proteins.py生成的OUT_DIR
OUT_DIR = os.path.realpath(args.out_dir)


TOMOS_DIR = OUT_DIR + f'/tomos'
TEM_DIR = OUT_DIR + f'/tem'


# 根据参数生成倾斜角度数组
TILT_ANGS = np.arange(args.tilt_start, args.tilt_end, args.tilt_step)

# 生成文件名后缀，包含倾斜角度信息
tilt_suffix = f"_tilt{args.tilt_start}to{args.tilt_end-1}step{args.tilt_step}"
tomo_den_out = TOMOS_DIR + "/tomo_den_refined" + ".mrc"

DETECTOR_SNR = None  # 0.2 # [.15, .25]
# MALIGN_MN = 1
# MALIGN_MX = 1.5
# MALIGN_SG = 0.2

VOI_VSIZE = 20 # 2.2 # A/vx

# 打印使用的参数
print(f"使用输出目录: {OUT_DIR}")
print(f"倾斜角度范围: {args.tilt_start} 到 {args.tilt_end-1}，步长: {args.tilt_step}")
print(f"倾斜角度数组: {TILT_ANGS}")

# TEM for 3D reconstructions
temic = tem.TEM(TEM_DIR)
vol = lio.load_mrc(tomo_den_out)
temic.gen_tilt_series_imod(vol, TILT_ANGS, ax="Y")
# temic.add_mics_misalignment(MALIGN_MN, MALIGN_MX, MALIGN_SG)
if DETECTOR_SNR is not None:
    if hasattr(DETECTOR_SNR, "__len__"):
        if len(DETECTOR_SNR) >= 2:
            snr = round(
                (DETECTOR_SNR[1] - DETECTOR_SNR[0]) * random.random()
                + DETECTOR_SNR[0],
                2,
            )
        else:
            snr = DETECTOR_SNR[0]
    else:
        snr = DETECTOR_SNR
    temic.add_detector_noise(snr)
temic.invert_mics_den()
temic.set_header(data="mics", p_size=(VOI_VSIZE, VOI_VSIZE, VOI_VSIZE))
temic.recon3D_imod()
temic.set_header(
    data="rec3d", p_size=(VOI_VSIZE, VOI_VSIZE, VOI_VSIZE), origin=(0, 0, 0)
)
if DETECTOR_SNR is not None:
    out_mics, out_tomo_rec = (
        TOMOS_DIR
        + "/tomo_mics_"
        + "refined"
        + tilt_suffix
        + "_snr"
        + str(snr)
        + ".mrc",
        TOMOS_DIR
        + "/tomo_rec_"
        + "refined"
        + tilt_suffix
        + "_snr"
        + str(snr)
        + ".mrc",
    )
else:
    out_mics, out_tomo_rec = (
        TOMOS_DIR + "/tomo_mics_" + "refined" + tilt_suffix + ".mrc",
        TOMOS_DIR + "/tomo_rec_" + "refined" + tilt_suffix + ".mrc",
    )
shutil.copyfile(TEM_DIR + "/out_micrographs.mrc", out_mics)
shutil.copyfile(TEM_DIR + "/out_rec3d.mrc", out_tomo_rec)
# synth_tomo.set_mics(out_mics)
# synth_tomo.set_tomo(out_tomo_rec)


print("Successfully terminated. (" + time.strftime("%c") + ")")