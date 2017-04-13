
import os
import numpy as np
from history import History

class DQNAgent(object):

    def __init__(self, num_actions, q_network, preproc, memory, policy, args):
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

    def train(self, env):
        self.update_target()

        print '########## burning in some steps #############'
        while self.memory.size() < self.args.burn_in_steps:

            history, state_mem, done = self.init_episode(env)
            act = self.policy['random'].select_action()
            for ep_iter in xrange(self.args.max_episode_length):
                if _every_not_0(ep_iter, self.args.action_steps):
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
        while iter_num <= self.args.train_steps:
            history, state_mem, done = self.init_episode(env)

            print '########## begin new episode #############'
            act = self.pick_action(state_mem, 'train', iter_num)
            for ep_iter in xrange(self.args.max_episode_length):
                if _every_not_0(ep_iter, self.args.action_steps):
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
                    self.train_online(iter_num)
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
            print '{:d} out of {:d} iterations'.format(iter_num, self.args.train_steps)

    def evaluate(self, env):
        total_reward = 0.0
        for episode in xrange(self.args.eval_episodes):
            history, state_mem, done = self.init_episode(env)

            episode_reward = 0.0
            act = self.pick_action(state_mem, 'eval')
            for ep_iter in xrange(self.args.max_episode_length):
                if _every_not_0(ep_iter, self.args.action_steps):
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
        average_reward = total_reward / self.args.eval_episodes
        print 'average episode reward: {:f}'.format(average_reward)

    def init_episode(self, env):
        env.reset()

        # construct a history object
        history = History(self.args.num_frames, self.args.action_steps)

        # begin each episode with 30 noop's
        act = 0
        for _ in xrange(self.args.num_init_frames):
            self.do_action(env, act, history)
        state_mem, _, done = history.get_next()
        return history, state_mem, done

    def do_action(self, env, act, history):
        frame, frame_reward, frame_done, _ = env.step(act)
        frame_mem = self.preproc.frame_to_frame_mem(frame)
        history.append(frame_mem, frame_reward, frame_done)

    def pick_action(self, state_mem, policy_type, iter_num=0):
        state = self.preproc.state_mem_to_state(state_mem)
        state = np.stack([state])
        q_online = self.q_network['online'].predict([state, self.null_act])[1]
        return self.policy[policy_type].select_action(q_online, iter_num)

    def train_online(self, iter_num):
        batch, b_idx, b_prob, b_state, b_act, b_state_next = self.get_batch()
        batch_wts = self.memory.get_batch_weights(b_idx, b_prob, iter_num)
        q_target_b, online = self.get_q_target(batch, b_state_next, b_act)
        q_online_b_act = self.q_network['online'].predict([b_state, b_act])[0]
        self.memory.update_priority(b_idx, q_target_b - q_online_b_act)
        online.train_on_batch([b_state, b_act], [q_target_b, self.null_target],
                              sample_weight=[batch_wts, batch_wts])

    def print_loss(self):
        batch, _, _, b_state, b_act, b_state_next = self.get_batch()
        q_target_b, _ = self.get_q_target(batch, b_state_next, b_act)
        loss_online = self.q_network['online'].evaluate([b_state, b_act],
            [q_target_b, self.null_target], verbose=0)
        loss_target = self.q_network['target'].evaluate([b_state, b_act],
            [q_target_b, self.null_target], verbose=0)
        print 'losses:', loss_online[0], loss_target[0]

    def update_target(self):
        print 'update update update update update'
        online_weights = self.q_network['online'].get_weights()
        self.q_network['target'].set_weights(online_weights)

    def get_batch(self):
        batch, b_idx, b_prob = self.memory.sample(self.args.batch_size)
        b_state = []
        b_act = []
        one_hot_eye = np.eye(self.num_actions, dtype=np.float32)
        b_state_next = []
        for st_m, act, rew, st_m_n, _ in batch:
            st = self.preproc.state_mem_to_state(st_m)
            b_state.append(st)
            b_act.append(one_hot_eye[act].copy())
            st_n = self.preproc.state_mem_to_state(st_m_n)
            b_state_next.append(st_n)
        b_state = np.stack(b_state)
        b_act = np.stack(b_act)
        b_state_next = np.stack(b_state_next)
        return batch, b_idx, b_prob, b_state, b_act, b_state_next

    def get_q_target(self, batch, b_state_next, b_act):
        if np.random.rand() < 0.5:
            online = self.q_network['target']
            target = self.q_network['online']
        else:
            online = self.q_network['online']
            target = self.q_network['target']
        q_online_b_n = online.predict([b_state_next, b_act])[1]
        q_target_b_n = target.predict([b_state_next, b_act])[1]
        q_target_b = []
        ziplist = zip(q_online_b_n, q_target_b_n, batch)
        for qon, qtn, (_, _, rew, _, db) in ziplist:
            full_reward = rew
            if not db:
                full_reward += self.args.discount * qon[np.argmax(qtn)]
            q_target_b.append([full_reward])
        return np.stack(q_target_b), online

def _every(iteration, interval):
    return not (iteration % interval)

def _every_not_0(iteration, interval):
    return not (iteration % interval) and iteration
