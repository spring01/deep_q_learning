#!/usr/bin/env python
"""Play some game with DQN."""

import gym
import argparse
from collections import defaultdict
from keras.optimizers import Adam
from dqn.dqn import DQN
from dqn.objectives import mean_huber_loss, null_loss
from dqn.policy import *
from dqn.history import History
from dqn.memory import PriorityMemory
from dqn.util import get_output_folder
from dqn.qnetwork import create_model
from dqn.preprocessors import *


def main():
    parser = argparse.ArgumentParser(description='Play a game with DQN')
    parser.add_argument('--env', default='Breakout-v0',
        help='Environment name')
    parser.add_argument('--output', default='output',
        help='Directory to save data to')
    parser.add_argument('--resize', nargs=2, type=int, default=(84, 84),
        help='Input shape')
    parser.add_argument('--num_frames', default=4, type=int,
        help='Number of frames in a state')
    parser.add_argument('--act_steps', default=4, type=int,
        help='Do an action for how many steps')
    parser.add_argument('--model_name', default='dqn', type=str,
        help='Model name')
    parser.add_argument('--show_preprocessing', default=False, type=bool,
        help='Show preprocessing effect at the beginning')
    parser.add_argument('--mode', default='train', type=str,
        help='Running mode; train or test')
    DQN.add_arguments(parser)
    PriorityMemory.add_arguments(parser)
    Policy.add_arguments(parser)

    # learning rate for the optimizer
    parser.add_argument('--learning_rate', default=1e-4, type=float,
        help='Learning rate')

    # checkpoint
    parser.add_argument('--read_weights', default=None, type=str,
        help='Read weights from file')
    parser.add_argument('--read_memory', default=None, type=str,
        help='Read memory from file')

    # parse arguments
    args = parser.parse_args()

    # add new environments here
    if args.env in ['FlappyBird-v0']:
        import gym_ple
    print '########## All arguments:', args
    args.resize = tuple(args.resize)

    # environment
    env = gym.make(args.env)
    num_act = env.action_space.n

    # preprocessor
    env_preproc = defaultdict(lambda: Preprocessor)
    env_preproc['Breakout-v0'] = BottomSquarePreprocessor
    env_preproc['MsPacman-v0'] = TopSquarePreprocessor
    env_preproc['FlappyBird-v0'] = TopSquarePreprocessor
    preproc = env_preproc[args.env](args.resize)

    # show preprocessing effect
    env.reset()
    if args.show_preprocessing:
        for _ in range(100):
            frame, _, _, _ = env.step(env.action_space.sample())
        preproc.show_effect(frame)
        env.reset()

    # online and target q networks
    state_shape = preproc.height, preproc.width, args.num_frames
    online = create_model(state_shape, num_act, args.model_name)
    target = create_model(state_shape, num_act, args.model_name)
    q_net = {'online': online, 'target': target}

    # history and memory
    history = History(args.num_frames, args.act_steps)
    memory = PriorityMemory(args, args.act_steps, args.train_steps)

    # initialization, train, evaluation policies
    policy_rand = RandomPolicy(num_act)
    policy_train = LinearDecayGreedyEpsPolicy(args)
    policy_eval = GreedyEpsPolicy(args)
    policy = {'rand': policy_rand, 'train': policy_train, 'eval': policy_eval}

    # construct and compile the dqn agent
    output = get_output_folder(args.output, args.env)
    agent = DQN(num_act, q_net, preproc, history, memory, policy, output, args)
    agent.compile([mean_huber_loss, null_loss], Adam(lr=args.learning_rate))

    # read weights/memory if requested
    if args.read_weights is not None:
        agent.q_net['online'].load_weights(args.read_weights)
    if args.read_memory is not None:
        agent.memory.load(args.read_memory)

    # running
    if args.mode == 'train':
        print '########## training #############'
        agent.train(env)
    elif args.mode == 'eval':
        print '########## evaluation #############'
        agent.evaluate(env)
    elif args.mode == 'rand':
        print '########## random #############'
        agent.random(env)



if __name__ == '__main__':
    main()

