#!/bin/bash

############### Train CryoDeRec Model ##################

CUDA_VISIBLE_DEVICES=2 python denoise3d_missingRec.py \
          --save-prefix ./model_training/ribosome/cryoderec_model \
          --save-interval 25 \
          --N-train 270 \
          --N-test 50 \
          -a ./training_simulated_data/train_ribosome/train_noisy \
          -b ./training_simulated_data/train_ribosome/train_sim_gt \
          -c 96 \
          -mw 60 \
          --criteria L2 \
          -p 32 \
          -o ./test_results \
          --num-epochs 2 \
          --num-workers 8 \
          -d -2 \
          --batch-size 2 \
          --masked-loss-weight 0.1 



############### Test CryoDeRec Model ##################

CUDA_VISIBLE_DEVICES=2 python denoise3d_missingRec.py \
         -o ./test_results/ribosome_pretrain  \
         -m  ./pre_trained/model_ribosome/model.sav \
         -s 96 \
         -d -2 \
         --patch-padding 32 \
         --batch-size 1 \
         ./test_data/EM10499_TS01.mrc


CUDA_VISIBLE_DEVICES=2 python denoise3d_missingRec.py \
         -o ./test_results/nucleosome_pretrain  \
         -m  ./pre_trained/model_nucleosome/model.sav \
         -s 96 \
         -d -2 \
         --patch-padding 32 \
         --batch-size 1 \
         ./test_data/nucleosome_20190120_44.rec

CUDA_VISIBLE_DEVICES=2 python denoise3d_missingRec.py \
         -o ./test_results/hiv_pretrain  \
         -m  ./pre_trained/model_hiv/model.sav \
         -s 96 \
         -d -2 \
         --patch-padding 32 \
         --batch-size 1 \
         ./test_data/EM10643_HIV.mrc
