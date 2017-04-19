#!/bin/bash

python -u run_dqn.py --env Breakout-v0 --output ./output --resize 84 110 --memory_steps 1000 --burn_in_steps 200 --model_name cheap_dueling_dqn --dense_size 128 --train_steps 2000 --target_reset_interval 500 --do_render T --eval_episodes 2 --num_frame 4 --save_interval 500 --episode_seed 15213

