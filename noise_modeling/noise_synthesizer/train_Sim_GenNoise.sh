

## train
python crSim_generate_noise_with_GAN.py \
       --model WGAN-GP \
       --is_train True \
       --dataroot ./ts_noise_patch \
       --dataset custom \
       --load_D ./pre_trained/discriminator.pkl \
       --load_G ./pre_trained/generator.pkl \
       --cuda True \
       --batch_size 128 \
       --tilt_angles 41 \
       --patch_size 256 \
       --image_size 400 \
       --save_path ./synthesized_noise_output




## inference
python crSim_generate_noise_with_GAN.py \
       --model WGAN-GP \
       --is_train False \
       --dataset custom \
       --load_D ./pre_trained/discriminator.pkl \
       --load_G ./pre_trained/generator.pkl \
       --cuda True \
       --batch_size 128 \
       --tilt_angles 41 \
       --patch_size 256 \
       --image_size 1024 \
       --save_path ./synthesized_noise_output