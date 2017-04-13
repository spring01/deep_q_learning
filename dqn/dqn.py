
import os
import numpy as np
from history import History

class DQNAgent(object):

    def __init__(self, num_actions, q_network, preproc, memory,
                 policy, args):
        self.num_actions = num_actions
        self.q_network = q_network
        self.preproc = preproc
        self.memory = memory
        self.policy = policy
        self.args = args
        self.null_act = np.zeros([1, self.num_actions])
        self.null_target = np.zeros([self.args.batch_size, self.num_actions])

    def compile(self, loss, optimizer):
        self.q_network['online'].compile(loss=loss, optimizer=optimizer)
        self.q_network['target'].compile(loss=loss, optimizer=optimizer)

    def fit(self, env):
        self.update_target()

        # filling in self.args.num_burn_in states
        print '########## burning in some samples #############'
        while len(self.memory) < self.args.size_burn_in:

            history, state_mem, done = self.init_episode(env)
            act = self.policy['random'].select_action()
            for ep_iter in xrange(self.args.max_episode_length):
                if _every_not_0(ep_iter, self.args.action_change_interval):
                    # current list in history is the next state
                    state_mem_next, reward, done = history.get_next()

                    # store transition into replay memory
                    transition = state_mem, act, reward, state_mem_next, done
                    self.memory.append(transition)
                    state_mem = state_mem_next

                    # get new action
                    act = self.policy['random'].select_action()

                # break if done
                if done:
                    break

                # do action to get the next state
                self.do_action(env, act, history)

        iter_num = 0
        eval_flag = False
        while iter_num <= self.args.num_train:
            history, state_mem, done = self.init_episode(env)

            print '########## begin new episode #############'
            act = self.pick_action(state_mem, 'train', iter_num)
            for ep_iter in xrange(self.args.max_episode_length):
                if _every_not_0(ep_iter, self.args.action_change_interval):
                    # current list in history is the next state
                    state_mem_next, reward, done = history.get_next()
                    reward = self.preproc.clip_reward(reward)

                    # store transition into replay memory
                    transition = state_mem, act, reward, state_mem_next, done
                    self.memory.append(transition)
                    state_mem = state_mem_next

                    # get online q value and get action
                    act = self.pick_action(state_mem, 'train', iter_num)

                # break if done
                if done:
                    break

                # do action to get the next state
                self.do_action(env, act, history)

                # update networks
                if _every(iter_num, self.args.online_train_interval):
                    self.train_online()
                if _every(iter_num, self.args.target_reset_interval):
                    self.update_target()

                # set evaluation flag
                if _every(iter_num, self.args.eval_interval):
                    eval_flag = True

                # save model
                if _every(iter_num, self.args.save_interval):
                    weights_save = os.path.join(self.args.output,
                        'online_{:d}.h5'.format(iter_num))
                    print '########## saving models and memory #############'
                    self.q_network['online'].save_weights(weights_save)
                    print 'online weights written to {:s}'.format(weights_save)
                    memory_save = os.path.join(self.args.output, 'memory.p')
                    self.memory.save(memory_save)
                    print 'replay memory written to {:s}'.format(memory_save)

                state_mem = state_mem_next

                if _every(iter_num, self.args.print_loss_interval):
                    self.print_loss()

                iter_num += 1
            # evaluation
            if eval_flag:
                eval_flag = False
                print '########## evaluation #############'
                self.evaluate(env)
            print '{:d} out of {:d} iterations'.format(iter_num, self.args.num_train)

    def evaluate(self, env):
        total_reward = 0.0
        for episode in xrange(self.args.eval_episodes):
            history, state_mem, done = self.init_episode(env)

            episode_reward = 0.0
            act = self.pick_action(state_mem, 'eval')
            for ep_iter in xrange(self.args.max_episode_length):
                if _every_not_0(ep_iter, self.args.action_change_interval):
                    # current list in history is the next state
                    state_mem, reward, done = history.get_next()
                    episode_reward += reward

                    # get online q value and get action
                    act = self.pick_action(state_mem, 'eval')

                # break if done
                if done:
                    break

                # do action to get the next state
                self.do_action(env, act, history)

            print '  episode reward: {:f}'.format(episode_reward)
            total_reward += episode_reward
            avg_reward = total_reward / self.args.eval_episodes
        print 'average episode reward: {:f}'.format(avg_reward)

    def init_episode(self, env):
        env.reset()

        # construct a history object
        history = History(self.args.num_frames, self.args.action_change_interval)

        # begin each episode with 30 noop's
        act = 0
        for _ in xrange(self.args.num_init_frames):
            frame, frame_rwd, frame_done, _ = env.step(act)
            frame_mem = self.preproc.frame_to_frame_mem(frame)
            history.append(frame_mem, frame_rwd, frame_done)
        state_mem, _, done = history.get_next()
        return history, state_mem, done

    def do_action(self, env, act, history):
        frame, frame_rwd, frame_done, _ = env.step(act)
        frame_mem = self.preproc.frame_to_frame_mem(frame)
        history.append(frame_mem, frame_rwd, frame_done)

    def pick_action(self, state_mem, policy_type, iter_num=0):
        state = self.preproc.state_mem_to_state(state_mem)
        state = np.stack([state])
        q_online = self.q_network['online'].predict([state, self.null_act])[1]
        return self.policy[policy_type].select_action(q_online, iter_num)

    def train_online(self):
        batch, input_b, act_b, input_b_n = self.get_batch()
        online, target = self.roll_online_target()
        q_target_b = self.get_q_target(batch, input_b_n, act_b, online, target)
        online.train_on_batch([input_b, act_b], [q_target_b, self.null_target])

    def print_loss(self):
        batch, input_b, act_b, input_b_n = self.get_batch()
        online, target = self.roll_online_target()
        q_target_b = self.get_q_target(batch, input_b_n, act_b, online, target)
        null_target = np.zeros(act_b.shape)
        loss_online = self.q_network['online'].evaluate([input_b, act_b],
            [q_target_b, null_target], verbose=0)
        loss_target = self.q_network['target'].evaluate([input_b, act_b],
            [q_target_b, null_target], verbose=0)
        print 'losses:', loss_online[0], loss_target[0]

    def update_target(self):
        print 'update update update update update'
        online_weights = self.q_network['online'].get_weights()
        self.q_network['target'].set_weights(online_weights)

    def get_batch(self):
        batch = self.memory.sample(self.args.batch_size)
        input_b = []
        act_b = []
        one_hot_eye = np.eye(self.num_actions, dtype=np.float32)
        input_b_n = []
        for st_m, act, rew, st_m_n, done_b in batch:
            st = self.preproc.state_mem_to_state(st_m)
            input_b.append(st)
            act_b.append(one_hot_eye[act].copy())
            st_n = self.preproc.state_mem_to_state(st_m_n)
            input_b_n.append(st_n)
        input_b = np.stack(input_b)
        act_b = np.stack(act_b)
        input_b_n = np.stack(input_b_n)
        return batch, input_b, act_b, input_b_n

    def roll_online_target(self):
        if np.random.rand() < 0.5:
            online = self.q_network['target']
            target = self.q_network['online']
        else:
            online = self.q_network['online']
            target = self.q_network['target']
        return online, target

    def get_q_target(self, batch, input_b_n, act_b, online, target):
        q_online_b_n = online.predict([input_b_n, act_b])[1]
        q_target_b_n = target.predict([input_b_n, act_b])[1]
        q_target_b = []
        ziplist = zip(q_online_b_n, q_target_b_n, batch)
        for qon, qtn, (_, _, rew, _, db) in ziplist:
            full_reward = rew
            if not db:
                full_reward += self.args.discount * qon[np.argmax(qtn)]
            q_target_b.append([full_reward])
        return np.stack(q_target_b)

def _every(iteration, interval):
    return not (iteration % interval)

def _every_not_0(iteration, interval):
    return not (iteration % interval) and iteration
