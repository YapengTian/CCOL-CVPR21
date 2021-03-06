#!/bin/bash

OPTS=""
OPTS+="--mode eval "
OPTS+="--id MUSIC_CCoL-4mix-LogFreq-resnet18dilated-unet7-linear-frames3stride1-maxpool-ratio-weightedLoss-channels32-epoch60-step30_50 "
OPTS+="--list_train ../data/Music/train.csv "
OPTS+="--list_val ../data/Music/test_sep.csv "

# Models
OPTS+="--arch_sound_ground vggish "
OPTS+="--arch_frame_ground resnet18 "
OPTS+="--arch_sound unet7 "
OPTS+="--arch_synthesizer linear "
OPTS+="--arch_grounding base "
OPTS+="--arch_frame resnet18dilated "
OPTS+="--img_pool maxpool "
OPTS+="--num_channels 32 "
OPTS+="--split test "

# loss
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
OPTS+="--audLen 65535 "
OPTS+="--audRate 11025 "

python -u main_silent.py $OPTS
