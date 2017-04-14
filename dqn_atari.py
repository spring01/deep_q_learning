#!/usr/bin/env python
"""Run Atari Environment with DQN."""

# common imports
import gym
import argparse
from dqn.dqn import DQNAgent
from dqn.objectives import mean_huber_loss, null_loss
from dqn.policy import *
from dqn.history import History
from dqn.memory import PriorityMemory
from dqn.util import get_output_folder
from dqn.qnetwork import create_model
from keras.optimizers import Adam

# game specific imports
from atari.preprocessor import AtariPreprocessor


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari games')
    parser.add_argument('--env', default='Breakout-v0',
        help='Atari env name')
    parser.add_argument('--output', default='atari-v0',
        help='Directory to save data to')
    parser.add_argument('--resize', nargs=2, type=int, default=(84, 110),
        help='Input shape')
    parser.add_argument('--num_frames', default=4, type=int,
        help='Number of frames in a state')
    parser.add_argument('--act_steps', default=4, type=int,
        help='Do an action for how many steps')
    parser.add_argument('--model_name', default='dqn', type=str,
        help='Model name')
    parser.add_argument('--mode', default='train', type=str,
        help='Running mode; train or test')

    DQNAgent.add_arguments(parser)
    PriorityMemory.add_arguments(parser)
    Policy.add_arguments(parser)

    # learning rate for the optimizer
    parser.add_argument('--learning_rate', default=1e-5, type=float,
        help='Learning rate alpha')

    # checkpoint
    parser.add_argument('--read_weights', default=None, type=str,
        help='Read weights from file')
    parser.add_argument('--read_memory', default=None, type=str,
        help='Read memory from file')

    args = parser.parse_args()
    print '########## All arguments:', args
    args.resize = tuple(args.resize)
    state_shape = args.resize[0], args.resize[0], args.num_frames
    output = get_output_folder(args.output, args.env)

    env = gym.make(args.env)
    num_act = env.action_space.n

    # online and target q networks
    online = create_model(state_shape, num_act, args.model_name)
    target = create_model(state_shape, num_act, args.model_name)
    q_net = {'online': online, 'target': target}

    # preprocessor, history, and memory
    preproc = AtariPreprocessor(args.resize)
    history = History(args.num_frames, args.act_steps)
    memory = PriorityMemory(args, args.act_steps, args.train_steps)

    # initialization, train, evaluation policies
    policy_init = RandomPolicy(num_act)
    policy_train = LinearDecayGreedyEpsPolicy(args)
    policy_eval = GreedyEpsPolicy(args)
    policy = {'init': policy_init, 'train': policy_train, 'eval': policy_eval}

    # construct and compile the dqn agent
    agent = DQNAgent(num_act, q_net, preproc, history, memory, policy, output,
                     args)
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
    elif args.mode == 'evaluation':
        print '########## evaluation #############'
        agent.evaluate(env)



if __name__ == '__main__':
    main()

