
import os
import numpy as np

class DQN(object):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--discount', default=0.99, type=float,
            help='Discount factor gamma')
        parser.add_argument('--num_init_frames', default=30, type=int,
            help='Number of initialization frames in an episode')
        parser.add_argument('--train_steps', default=10000000, type=int,
            help='Number of training sampled interactions with the environment')
        parser.add_argument('--max_episode_length', default=100000, type=int,
            help='Maximum length of an episode')
        parser.add_argument('--batch_size', default=32, type=int,
            help='How many samples in each minibatch')
        parser.add_argument('--online_train_interval', default=16, type=int,
            help='Interval to train the online network')
        parser.add_argument('--target_reset_interval', default=160000, type=int,
            help='Interval to reset the target network')
        parser.add_argument('--print_loss_interval', default=1000, type=int,
            help='Interval to print losses')
        parser.add_argument('--save_interval', default=1000000, type=int,
            help='Interval to save weights and memory')
        parser.add_argument('--eval_interval', default=100000, type=int,
            help='Evaluation interval')
        parser.add_argument('--eval_episodes', default=20, type=int,
            help='Number of episodes in evaluation')
        parser.add_argument('--do_render', default=False, type=bool,
            help='Do rendering or not')
        parser.add_argument('--episode_seed', default=None, type=int,
            help='Setting seed to get the same episode each time')

    def __init__(self, num_act, q_net, preproc, history, memory, policy, output,
                 args):
        self.q_net = q_net
        self.preproc = preproc
        self.history = history
        self.memory = memory
        self.policy = policy
        self.output = output
        self.discount = args.discount
        self.num_init_frames = args.num_init_frames
        self.train_steps = args.train_steps
        self.max_episode_length = args.max_episode_length
        self.batch_size = args.batch_size
        self.online_train_interval = args.online_train_interval
        self.target_reset_interval = args.target_reset_interval
        self.print_loss_interval = args.print_loss_interval
        self.save_interval = args.save_interval
        self.eval_interval = args.eval_interval
        self.eval_episodes = args.eval_episodes
        self.do_render = args.do_render
        self.episode_seed = args.episode_seed
        self.null_act = np.zeros([1, num_act])
        self.null_target = np.zeros([self.batch_size, num_act])
        self.one_hot_act = np.eye(num_act, dtype=np.float32)

    def compile(self, loss, optimizer):
        self.q_net['online'].compile(loss=loss, optimizer=optimizer)
        self.q_net['target'].compile(loss=loss, optimizer=optimizer)

    def random(self, env):
        total_reward = 0.0
        for episode in xrange(self.eval_episodes):
            episode_reward = self.run_episode(env, 'rand', 0, store=False)[0]
            print '  random episode reward: {:f}'.format(episode_reward)
            total_reward += episode_reward
        average_reward = total_reward / self.eval_episodes
        print 'random average episode reward: {:f}'.format(average_reward)

    def train(self, env):
        self.update_target()

        print '########## burning in some steps #############'
        while self.memory.size() < self.memory.burn_in_steps:
            self.run_episode(env, 'eval', 0)
            self.memory.print_status()

        print '########## begin training #############'
        iter_num = 0
        num_episodes = 0
        while iter_num <= self.train_steps:
            _, iter_num, eval_flag = self.run_episode(env, 'train', iter_num)
            num_episodes += 1
            print '  iter {:d} out of {:d}'.format(iter_num, self.train_steps)
            print '  number of episodes: {:d}'.format(num_episodes)
            self.memory.print_status()
            if eval_flag:
                print '########## evaluation #############'
                self.evaluate(env)


    def evaluate(self, env):
        total_reward = 0.0
        for episode in xrange(self.eval_episodes):
            episode_reward = self.run_episode(env, 'eval', 0, store=False)[0]
            print '  episode reward: {:f}'.format(episode_reward)
            total_reward += episode_reward
        average_reward = total_reward / self.eval_episodes
        print 'average episode reward: {:f}'.format(average_reward)

    def run_episode(self, env, mode, iter_num, store=True):
        print '*** New episode with mode:', mode
        policy = self.policy[mode]
        state_mem, done = self.init_episode(env)
        episode_reward = 0.0
        eval_flag = False
        act = self.pick_action(state_mem, policy, iter_num)
        frame_done = False
        for ep_iter in xrange(self.max_episode_length):
            if _every_not_0(ep_iter, self.history.act_steps):
                # current list in history is the next state
                state_mem_next, reward, done = self.history.get_next()
                episode_reward += reward
                reward = self.preproc.clip_reward(reward)

                # store transition into replay memory
                if store:
                    transition = state_mem, act, reward, state_mem_next, done
                    self.memory.append(transition)
                state_mem = state_mem_next

                # get online q value and get action
                act = self.pick_action(state_mem, policy, iter_num)

            # break if done
            if done:
                break

            # do action to get the next state
            if not frame_done:
                frame_done = self.do_action(env, act)

            # modify eval_flag and iter_num
            if mode == 'train':
                eval_flag = self.extra_work_train(iter_num) or eval_flag
            iter_num += 1
        print '*** End of episode'
        return episode_reward, iter_num, eval_flag

    def extra_work_train(self, iter_num):
        # update networks
        if _every(iter_num, self.online_train_interval):
            self.train_online(iter_num)
        if _every(iter_num, self.target_reset_interval):
            self.update_target()

        # save model
        if _every(iter_num, self.save_interval):
            weights_save = os.path.join(self.output,
                'online_{:d}.h5'.format(iter_num))
            print '########## saving models and memory #############'
            self.q_net['online'].save_weights(weights_save)
            print 'online weights written to {:s}'.format(weights_save)
            memory_save = os.path.join(self.output, 'memory.p')
            self.memory.save(memory_save)
            print 'replay memory written to {:s}'.format(memory_save)

        # print losses
        if _every(iter_num, self.print_loss_interval):
            self.print_loss()

        # return evaluation flag
        return _every(iter_num, self.eval_interval)

    def init_episode(self, env):
        # reset env and history
        env.reset()
        if self.episode_seed is not None:
            env.seed(self.episode_seed)
        self.history.reset()

        # begin each episode with noop's
        act = 0
        for _ in xrange(self.num_init_frames):
            self.do_action(env, act)
        state_mem, _, done = self.history.get_next()
        return state_mem, done

    def do_action(self, env, act):
        frame, frame_reward, frame_done, _ = env.step(act)
        if self.do_render:
            env.render()
        frame_mem = self.preproc.frame_to_frame_mem(frame)
        self.history.append(frame_mem, frame_reward, frame_done)
        return frame_done

    def pick_action(self, state_mem, policy, iter_num=0):
        state = self.preproc.state_mem_to_state(state_mem)
        state = np.stack([state])
        q_online = self.q_net['online'].predict([state, self.null_act])[1]
        return policy.select_action(q_online, iter_num)

    def train_online(self, iter_num):
        batch, b_idx, b_prob, b_state, b_act, b_state_next = self.get_batch()
        batch_wts = self.memory.get_batch_weights(b_idx, b_prob, iter_num)
        q_target_b, online = self.get_q_target(batch, b_state_next, b_act)
        q_online_b_act = self.q_net['online'].predict([b_state, b_act])[0]
        self.memory.update_priority(b_idx, q_target_b - q_online_b_act)
        online.train_on_batch([b_state, b_act], [q_target_b, self.null_target],
                              sample_weight=[batch_wts, batch_wts])

    def print_loss(self):
        batch, _, _, b_state, b_act, b_state_next = self.get_batch()
        q_target_b, _ = self.get_q_target(batch, b_state_next, b_act)
        loss_online = self.q_net['online'].evaluate([b_state, b_act],
            [q_target_b, self.null_target], verbose=0)
        loss_target = self.q_net['target'].evaluate([b_state, b_act],
            [q_target_b, self.null_target], verbose=0)
        print 'losses:', loss_online[0], loss_target[0]

    def update_target(self):
        print 'update update update update update'
        online_weights = self.q_net['online'].get_weights()
        self.q_net['target'].set_weights(online_weights)

    def get_batch(self):
        batch, b_idx, b_prob = self.memory.sample(self.batch_size)
        b_state = []
        b_act = []
        b_state_next = []
        for st_m, act, rew, st_m_n, _ in batch:
            st = self.preproc.state_mem_to_state(st_m)
            b_state.append(st)
            b_act.append(self.one_hot_act[act].copy())
            st_n = self.preproc.state_mem_to_state(st_m_n)
            b_state_next.append(st_n)
        b_state = np.stack(b_state)
        b_act = np.stack(b_act)
        b_state_next = np.stack(b_state_next)
        return batch, b_idx, b_prob, b_state, b_act, b_state_next

    def get_q_target(self, batch, b_state_next, b_act):
        if np.random.rand() < 0.5:
            online = self.q_net['target']
            target = self.q_net['online']
        else:
            online = self.q_net['online']
            target = self.q_net['target']
        q_online_b_n = online.predict([b_state_next, b_act])[1]
        q_target_b_n = target.predict([b_state_next, b_act])[1]
        q_target_b = []
        ziplist = zip(q_online_b_n, q_target_b_n, batch)
        for qon, qtn, (_, _, rew, _, db) in ziplist:
            full_reward = rew
            if not db:
                full_reward += self.discount * qon[np.argmax(qtn)]
            q_target_b.append([full_reward])
        return np.stack(q_target_b), online

def _every(iter_num, interval):
    return not (iter_num % interval)

def _every_not_0(iter_num, interval):
    return not (iter_num % interval) and iter_num
