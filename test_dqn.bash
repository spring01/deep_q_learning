#!/bin/bash

python -u dqn_atari.py --env Breakout-v0 --output ./output --input_shape 84 110 --replay_buffer_size 1000 --num_burn_in 200 --model_name cheap_dqn --num_train 2000 --target_reset_interval 500 --do_render T --eval_episodes 2 --num_frame 4 --save_interval 500

