"""
Script for generating tomograms simulating all features available
monomers
    Input:
        - Number of tomograms to simulate
        - Tomogram dimensions parameter
        - Tomogram maximum occupancy
        - Features to simulate:
            + Membranes
            + Polymers:
                + Helicoidal fibers
                + Globular protein clusters
        - 3D reconstruction paramaters
    Output:
        - The simulated density maps
        - The 3D reconstructed tomograms
        - Micrograph stacks
        - Polydata files
        - STAR file mapping particle coordinates and orientations with tomograms
"""

__author__ = 'Antonio Martinez-Sanchez'

import os
import sys
import time
import random
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
    """Parse command-line arguments for simulation output directory."""
    parser = argparse.ArgumentParser(description='Generate synthetic tomograms with all protein features.')
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help=(
            'Output directory for generated data '
            '(e.g. data_generated/polnet_test_20260304_emd11999_bin8_1024). '
            'If not set, the hard-coded default in this script is used.'
        ),
    )
    return parser.parse_args()


args = parse_arguments()

# Common tomogram settings
ROOT_PATH = os.path.realpath(os.getcwd() + '/data')
NTOMOS = 1 # 12
# VOI_SHAPE = (400, 400, 236) # (400, 400, 236) # vx or a path to a mask (1-foreground, 0-background) tomogram
# VOI_OFFS =  ((4,396), (4,396), (28,218)) # ((4,396), (4,396), (4,232)) # ((4,1852), (4,1852), (32,432)) # ((4,1852), (4,1852), (4,232)) # vx

VOI_SHAPE = (1024, 1024, 400) # (400, 400, 236) # vx or a path to a mask (1-foreground, 0-background) tomogram
VOI_OFFS =  ((24,1000), (24,1000), (50,350)) # ((4,396), (4,396), (28,218)) # ((4,396), (4,396), (4,232)) # ((4,1852), (4,1852), (32,432)) # ((4,1852), (4,1852), (4,232)) # vx
VOI_VSIZE = 20 # 2.2 # A/vx
GTRUTH_POINTS_RAD = 35 # nm
MMER_TRIES = 50
PMER_TRIES = 100

# Lists with the features to simulate
PROTEINS_LIST = ['empiar_10499_pns/emd11999_bin8.pns']


# Proportions list, specifies the proportion for each protein, this proportion is tried to be achieved but no guaranteed
# The toal sum of this list must be 1
PROP_LIST = None # [.4, .6]
if PROP_LIST is not None:
    assert sum(PROP_LIST) == 1

# DIST_OFF = 5 # A / vx
SURF_DEC = 0.9 # Target reduction factor for surface decimation (default None)

if args.out_dir is not None:
    OUT_DIR = os.path.realpath(args.out_dir)
else:
    OUT_DIR = os.path.realpath(ROOT_PATH + '/../data_generated/polnet_test_20260304_emd11999_bin8_1024')
os.makedirs(OUT_DIR, exist_ok=True)

TEM_DIR = OUT_DIR + '/tem'
TOMOS_DIR = OUT_DIR + '/tomos'
# print(TEM_DIR)
# exit(0)
os.makedirs(TOMOS_DIR, exist_ok=True)
os.makedirs(TEM_DIR, exist_ok=True)

# OUTPUT LABELS
LBL_MB = 1
LBL_AC = 2
LBL_MT = 3
LBL_CP = 4
LBL_MP = 5
# LBL_BR = 6

##### Main procedure

set_stomos = SetTomos()
vx_um3 = (VOI_VSIZE * 1e-4) ** 3

# Preparing intermediate directories
clean_dir(TEM_DIR)
clean_dir(TOMOS_DIR)

# Loop for tomograms
for tomod_id in range(NTOMOS):

    print('GENERATING TOMOGRAM NUMBER:', tomod_id)
    hold_time = time.time()

    # Generate the VOI and tomogram density
    if isinstance(VOI_SHAPE, str):
        voi = lio.load_mrc(VOI_SHAPE) > 0
        voi_off = np.zeros(shape=voi.shape, dtype=bool)
        voi_off[VOI_OFFS[0][0]:VOI_OFFS[0][1], VOI_OFFS[1][0]:VOI_OFFS[1][1], VOI_OFFS[2][0]:VOI_OFFS[2][1]] = True
        voi = np.logical_and(voi, voi_off)
        del voi_off
    else:
        voi = np.zeros(shape=VOI_SHAPE, dtype=bool)
        voi[VOI_OFFS[0][0]:VOI_OFFS[0][1], VOI_OFFS[1][0]:VOI_OFFS[1][1], VOI_OFFS[2][0]:VOI_OFFS[2][1]] = True
        voi_inital_invert = np.invert(voi)
    voi_voxels = voi.sum()
    tomo_lbls = np.zeros(shape=VOI_SHAPE, dtype=np.float32)
    tomo_den = np.zeros(shape=voi.shape, dtype=np.float32)
    synth_tomo = SynthTomo()
    poly_vtp, mbs_vtp, skel_vtp = None, None, None
    entity_id = 1
    cp_voxels = 0
    set_mbs = None

    count_prots = 0
    model_surfs, models, model_masks, model_codes = list(), list(), list(), list()
    # Loop for the list of input proteins loop
    for p_id, p_file in enumerate(PROTEINS_LIST):

        print('\tPROCESSING FILE:', p_file)

        # Loading the protein
        protein = MmerFile(ROOT_PATH + '/' + p_file)

        # Generating the occupancy
        hold_occ = protein.get_pmer_occ()
        if hasattr(hold_occ, '__len__'):
            hold_occ = OccGen(hold_occ).gen_occupancy()

        # Genrate the SAWLC network associated to the input protein
        # Polymer parameters
        # To read macromolecular models first we try to find the absolute path and secondly the relative to ROOT_PATH
        try:
            model = lio.load_mrc(protein.get_mmer_svol())
        except FileNotFoundError:
            model = lio.load_mrc(ROOT_PATH + '/' + protein.get_mmer_svol())
        # model = lio.load_mrc(ROOT_PATH + '/' + protein.get_mmer_svol())
        model = lin_map(model, lb=0, ub=1)
        model = vol_cube(model)
        model_mask = model < protein.get_iso()
        model[model_mask] = 0
        model_surf = pp.iso_surface(model, protein.get_iso(), closed=False, normals=None)
        if SURF_DEC is not None:
            model_surf = pp.poly_decimate(model_surf, SURF_DEC)
        center = .5 * np.asarray(model.shape, dtype=float)
        # Monomer centering
        model_surf = pp.poly_translate(model_surf, -center)
        # Voxel resolution scaling
        model_surf = pp.poly_scale(model_surf, VOI_VSIZE)
        model_surfs.append(model_surf)
        surf_diam = pp.poly_diam(model_surf) * protein.get_pmer_l()
        models.append(model)
        model_masks.append(model_mask)
        model_codes.append(protein.get_mmer_id())

        # Network generation
        pol_l_generator = PGenHelixFiber()
        if PROP_LIST is None:
            pol_s_generator = SGenUniform()
        else:
            assert len(PROP_LIST) == len(PROTEINS_LIST)
            pol_s_generator = SGenProp(PROP_LIST)
        net_sawlc = NetSAWLC(voi, VOI_VSIZE, protein.get_pmer_l() * surf_diam, model_surf, protein.get_pmer_l_max(),
                             pol_l_generator, hold_occ, protein.get_pmer_over_tol(), poly=None,
                             svol=model < protein.get_iso(), tries_mmer=MMER_TRIES, tries_pmer=PMER_TRIES)
        net_sawlc.build_network()

        # Density tomogram updating
        net_sawlc.insert_density_svol(model_mask, voi, VOI_VSIZE, merge='min')
        net_sawlc.insert_density_svol(model, tomo_den, VOI_VSIZE, merge='max')
        hold_lbls = np.zeros(shape=tomo_lbls.shape, dtype=np.float32)
        net_sawlc.insert_density_svol(np.invert(model_mask), hold_lbls, VOI_VSIZE, merge='max')
        tomo_lbls[hold_lbls > 0] = entity_id
        count_prots += net_sawlc.get_num_mmers()
        cp_voxels += (tomo_lbls == entity_id).sum()
        hold_vtp = net_sawlc.get_vtp()
        hold_skel_vtp = net_sawlc.get_skel()
        pp.add_label_to_poly(hold_vtp, entity_id, 'Entity', mode='both')
        pp.add_label_to_poly(hold_skel_vtp, entity_id, 'Entity', mode='both')
        pp.add_label_to_poly(hold_vtp, LBL_CP, 'Type', mode='both')
        pp.add_label_to_poly(hold_skel_vtp, LBL_CP, 'Type', mode='both')
        if poly_vtp is None:
            poly_vtp = hold_vtp
            skel_vtp = hold_skel_vtp
        else:
            poly_vtp = pp.merge_polys(poly_vtp, hold_vtp)
            skel_vtp = pp.merge_polys(skel_vtp, hold_skel_vtp)
        synth_tomo.add_network(net_sawlc, 'SAWLC', entity_id, code=protein.get_mmer_id())
        entity_id += 1


    # Storing simulated density results
    tomo_den_out = TOMOS_DIR + '/tomo_den_' + str(tomod_id) + '.mrc'
    lio.write_mrc(tomo_den, tomo_den_out, v_size=VOI_VSIZE)
    synth_tomo.set_den(tomo_den_out)
    tomo_lbls_out = TOMOS_DIR + '/tomo_lbls_' + str(tomod_id) + '.mrc'
    lio.write_mrc(tomo_lbls, tomo_lbls_out, v_size=VOI_VSIZE)

    print('\t\t-TOMOGRAM', str(tomod_id), 'DENSITY STATISTICS:')
    print('\t\t\t+Proteins:', count_prots, '#, ', cp_voxels * vx_um3, 'um**3, ', 100. * (cp_voxels / voi_voxels), '%')
    print('\t\t\t+Time for generation: ', (time.time() - hold_time) / 60, 'mins')

# Storing tomograms CSV file
set_stomos.save_csv(OUT_DIR + '/tomos_motif_list.csv')

print('Successfully terminated. (' + time.strftime("%c") + ')')

