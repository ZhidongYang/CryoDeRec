#!/bin/bash

# --- 提取出现次数 >= 3 次的变量 ---
TOMO_DIR="../tomogram_simulation/data_generated/polnet_test_20260304_emd11999_bin8_1024"
WORK_DIR="./empiar_10499_data/polnet_test_20260304_emd11999_bin8_1024"
SNR="0.3"
M_VAL="0.3"
# ----------------------------------

echo "Step 1: Apply CTF"
python crSim_apply_ctf_to_tilts.py \
    -t ${TOMO_DIR}/tomos/tomo_mics_refined_tilt-60to62step3.mrc \
    -o ${WORK_DIR}/tomo_mics_refined_tilt-60to62step3_applied_ctf.mrc \
    -d1 1000 -d2 1300 -an 1 -kv 300 -ac 0.1 -cs 2.7 || { echo "Step 1: Apply CTF failed"; exit 1; }

echo "Step 2: Add noise"
python crSim_apply_noise_to_tilts_snr.py \
    -i ${WORK_DIR}/tomo_mics_refined_tilt-60to62step3_applied_ctf.mrc \
    -o ${WORK_DIR}/tomo_mics_refined_tilt-60to62step3_applied_ctf_noise_snr${SNR}.mrc \
    -s ${SNR} -k 0.15 \
    -n ./noise_synthesizer/synthesized_noise_output/noise_tilts_1024_1024_41.mrc || { echo "Step 2: Add noise failed"; exit 1; }

echo "Step 3: tomo3d reconstruction m=${M_VAL}"
./tomo3d -a ./tlt/out_tangs_60_60_3.tlt \
    -i ${WORK_DIR}/tomo_mics_refined_tilt-60to62step3_applied_ctf_noise_snr${SNR}.mrc \
    -o ${WORK_DIR}/final_tomogram_m_${M_VAL}.mrc \
    -n -m ${M_VAL} -z 400 || { echo "Step 3: tomo3d m=${M_VAL} failed"; exit 1; }

echo "Step 4: clip rotx m=${M_VAL}"
clip rotx ${WORK_DIR}/final_tomogram_m_${M_VAL}.mrc ${WORK_DIR}/final_tomogram_m_${M_VAL}_rotx.mrc || { echo "Step 4: clip rotx m=${M_VAL} failed"; exit 1; }

echo "All steps completed!"