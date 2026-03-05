#!/bin/bash

set -e

#############################################
# Common configuration
#############################################

# Root output directory for step1 (synthetic tomograms)
OUT_DIR_SIM="data_generated/polnet_test_20260304_emd11999_bin8_1024"
PROTEIN_MRC="data/empiar_10499_pns/bin8_emd_11999.mrc"


echo "===== Tomogram simulation pipeline started ====="
echo "Using OUT_DIR_SIM=${OUT_DIR_SIM}"
echo

#############################################
# Step 1: Synthetic tomograms
#############################################
echo "[Step 1] Generating synthetic tomograms with all protein features..."
python step1_all_features_of_proteins.py \
  --out_dir "${OUT_DIR_SIM}" \
  || { echo "[Step 1] Generation failed"; exit 1; }
echo "[Step 1] Done."
echo

#############################################
# Step 2: Refine simulated density mask
#############################################
echo "[Step 2] Refining simulated density mask (merge membranes and proteins)..."
echo "         proteins : ${PROTEIN_MRC}"
python step2_refine_simulated_dentsity_mask.py \
    -m "${OUT_DIR_SIM}/tomos/tomo_den_0.mrc" \
    -p "${PROTEIN_MRC}" \
    -o "${OUT_DIR_SIM}/tomos/tomo_den_refined.mrc" \
    || { echo "[Step 2] Refinement failed"; exit 1; }
echo "[Step 2] Done."
echo


#############################################
# Step 3a/3b: 3D reconstruction with different tilt ranges
#############################################


echo "[Step 3a] 3D reconstruction (limited tilt range: -60 to 62, step 3)..."
python step3_recon3d.py \
    --out_dir "${OUT_DIR_SIM}" \
    --tilt_start -60 --tilt_end 63 --tilt_step 3 \
    || { echo "[Step 3a] Reconstruction failed"; exit 1; }
echo "[Step 3a] Done."
echo


echo "[Step 3b] 3D reconstruction (full tilt range: -90 to 90, step 1)..."
python step3_recon3d.py \
    --out_dir "${OUT_DIR_SIM}" \
    --tilt_start -90 --tilt_end 91 --tilt_step 1 \
    || { echo "[Step 3b] Reconstruction failed"; exit 1; }
echo "[Step 3b] Done."
echo

#############################################
# Step 4: tomo3d reconstruction and rotation
#############################################
echo "[Step 4] tomo3d reconstruction..."
./tomo3d -a "${OUT_DIR_SIM}/tem/out_tangs.tlt" \
    -i "${OUT_DIR_SIM}/tomos/tomo_mics_refined_tilt-90to90step1.mrc" \
    -o "${OUT_DIR_SIM}/tomos/tomo_rec_refined_tilt-90to90step1_tomo3d.mrc" \
    -n -z 400 \
    || { echo "[Step 4] tomo3d failed"; exit 1; }
echo "[Step 4] tomo3d reconstruction done."

echo "[Step 4] Rotating reconstructed tomogram (clip rotx)..."
clip rotx \
    "${OUT_DIR_SIM}/tomos/tomo_rec_refined_tilt-90to90step1_tomo3d.mrc" \
    "${OUT_DIR_SIM}/tomos/tomo_rec_refined_tilt-90to90step1_tomo3d_rox.mrc" \
    || { echo "[Step 4] clip rotx failed"; exit 1; }
echo "[Step 4] Rotation done."
echo

echo "===== All pipeline steps completed successfully ====="
