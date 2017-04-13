
class History(object):

    def __init__(self, num_frames, action_change_interval):
        self.obs_list = [None for _ in xrange(num_frames)]
        self.state_diff_frames = action_change_interval

    def append(self, frame_mem, frame_reward, frame_done):
        self.obs_list.append((frame_mem, frame_reward, frame_done))
        self.obs_list.pop(0)

    def get_next(self):
        state_mem_list = [tup[0] for tup in self.obs_list]
        reward = sum(tup[1] for tup in self.obs_list[-self.state_diff_frames:])
        done = self.obs_list[-1][2]
        return state_mem_list, reward, done
