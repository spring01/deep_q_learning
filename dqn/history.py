
class History(object):

    def __init__(self, num_frames, act_steps):
        self.num_frames = num_frames
        self.act_steps = act_steps
        self.reset()

    def append(self, frame_mem, frame_reward, frame_done):
        self.frame_list.append((frame_mem, frame_reward, frame_done))
        self.frame_list.pop(0)

    def get_next(self):
        state_mem_list = [tup[0] for tup in self.frame_list]
        reward = sum(tup[1] for tup in self.frame_list[-self.act_steps:])
        done = self.frame_list[-1][2]
        return state_mem_list, reward, done

    def reset(self):
        self.frame_list = [None for _ in xrange(self.num_frames)]
