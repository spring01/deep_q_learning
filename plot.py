
import sys
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'size': 18}
matplotlib.rc('font', **font)


def average_reward(filename):
    avg_reward = []
    with open(filename) as out:
        for line in out:
            if 'average' in line:
                avg_reward.append(float(line.strip().split()[-1]))
    return avg_reward

def smoothing(avg_reward):
    window = 20
    smooth_avg_reward = []
    for i in range(len(avg_reward)):
        smoothing_start = max(0, i - window)
        smoothing_end = min(len(avg_reward), i + window)
        smooth_avg_reward.append(np.mean(avg_reward[smoothing_start:smoothing_end]))
    return smooth_avg_reward


smooth_avg_reward_list = []
outname_list = []
for outname in sys.argv[1:]:
    avg_reward = average_reward(glob.glob(outname)[0])
    smooth_avg_reward_list.append(smoothing(avg_reward))
    outname_list.append(outname)

min_point = min([sm[0] for sm in smooth_avg_reward_list])

offset_list = []
for sm in smooth_avg_reward_list:
    offset_list.append(sm[0] - min_point)

smooth_avg_reward_offset_list = []
for i in range(len(smooth_avg_reward_list)):
    curve = [pt - offset_list[i] for pt in smooth_avg_reward_list[i]]
    plt.plot(curve, label=outname_list[i])

plt.legend()
plt.xlabel('Training frames (unit: 100,000 frames)')
plt.ylabel('Average episode reward')
plt.gcf().subplots_adjust(left=0.2)
plt.gcf().subplots_adjust(bottom=0.15)
plt.show()
#~ plt.savefig(game + '_learning_curve.pdf')

