import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

result_path = './result_tard'
if not os.path.exists(result_path):
    os.makedirs(result_path)


def cal_KPI(event_tracer, episode):
    # Due Date
    event_dd = event_tracer[event_tracer["Event"] == "Completed"]
    tard_earl_list = list(event_dd["Memo"])
    tard_earl_list = [int(dif) for dif in tard_earl_list]
    tard_earl_list.sort()
    plt.hist(tard_earl_list, bins=max(tard_earl_list) - min(tard_earl_list) - 1)
    plt.xlabel("Earliness <--| |--> Tardiness")
    plt.ylim([0, 300])
    plt.title("Episode 50")
    plt.savefig(result_path + '/episode {0}.png'.format(episode), dpi=300)
    plt.close()


if __name__ == "__main__":
    file_list = os.listdir('./result')
    episode_list = [int(file_name.split('.')[0].split('_')[1]) for file_name in file_list]
    episode_list.sort()

    setup_list = list()
    for episode in episode_list:
        event = pd.read_csv('./result/result_{0}.csv'.format(episode))
        cal_KPI(event, episode)
