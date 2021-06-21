#!/bin/bash

OPTS=""
OPTS+="--id MUSIC_GRD_Real_min "
OPTS+="--list_train /home/cxu-serve/p1/ytian21/dat/AVSS_data/MUSIC_dataset/data/train.csv "
OPTS+="--list_val /home/cxu-serve/p1/ytian21/dat/AVSS_data/MUSIC_dataset/data/val.csv "

# Models
OPTS+="--arch_sound_ground vggish "
OPTS+="--arch_frame_ground resnet18 "
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_grounding base "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "

# weights
#OPTS+="--weights_sound_ground  /home/cxu-serve/p1/ytian21/dat/AVSS_data/MUSIC_dataset/data/ckpt/MUSIC_GRD-4mix/sound_ground_best.pth "
#OPTS+="--weights_frame_ground  /home/cxu-serve/p1/ytian21/dat/AVSS_data/MUSIC_dataset/data/ckpt/MUSIC_GRD-4mix/frame_ground_best.pth "
#OPTS+="--weights_grounding /home/cxu-serve/p1/ytian21/dat/AVSS_data/MUSIC_dataset/data/ckpt/MUSIC_GRD-4mix/grounding_best.pth "


# binary mask, BCE loss, weighted loss
OPTS+="--binary_mask 0 "
OPTS+="--loss l1 "
OPTS+="--weighted_loss 1 "

# logscale in frequency
OPTS+="--num_mix 4 "
OPTS+="--log_freq 1 "

# frames-related
OPTS+="--num_frames 3 "
OPTS+="--stride_frames 1 "
OPTS+="--frameRate 1 "

# audio-related
OPTS+="--audLen 65535 " # 65535
OPTS+="--audRate 11025 " #11025

# learning params
OPTS+="--num_gpus 3 "
OPTS+="--workers 32 "
OPTS+="--batch_size_per_gpu 12 "
OPTS+="--lr_sound_ground 1e-4 "
OPTS+="--lr_frame_ground 1e-4 "
OPTS+="--lr_grounding 1e-4 "
OPTS+="--num_epoch 40 "
OPTS+="--lr_steps 20 30 "

# display, viz
OPTS+="--disp_iter 20 "
OPTS+="--num_vis 40 "
OPTS+="--num_val 256 "

python -u main.py $OPTS
