import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def smoothing(target_list, alpha=0.6):
    smoothing_list = [target_list[0]]
    for i in range(1, len(target_list)):
        l_t = alpha * target_list[i] + (1 - alpha) * smoothing_list[-1]
        smoothing_list.append(l_t)

    return smoothing_list


if __name__ == "__main__":
    smoothing_factor = 0.1

    file_list = os.listdir("./result")
    episode_list = [int(filename.split('.')[0].split("_")[1]) for filename in file_list]
    episode_list.sort()

    setup_list = list()
    tardiness_list = list()
    earliness_list = list()
    on_time_list = list()

    for episode in episode_list:
        event = pd.read_csv("./result/result_{0}.csv".format(episode))
        event_setup = event[event["Event"] == "Set-Up"]
        setup = len(event_setup)

        event_completed = event[event["Event"] == "Completed"]
        num_jobs = len(event_completed)
        difference_list = list(event_completed["Memo"])
        difference_list = [int(difference) for difference in difference_list]

        tardiness = [difference for difference in difference_list if difference > 0]
        earliness = [-difference for difference in difference_list if difference < 0]
        on_time = [difference for difference in difference_list if difference == 0]

        setup_list.append(round(setup / num_jobs, 2))
        tardiness_list.append(np.mean(tardiness))
        earliness_list.append(np.mean(earliness))
        on_time_list.append(round(len(on_time) / num_jobs, 2))

    plt.plot(episode_list, setup_list, alpha=0.2, color='purple')
    plt.plot(episode_list, smoothing(setup_list, alpha=smoothing_factor), color='purple')
    plt.ylim([0, 1])
    plt.title("Setup Ratio")

    plt.show()

    plt.plot(episode_list, tardiness_list, alpha=0.2, color="red")
    plt.plot(episode_list, smoothing(tardiness_list, alpha=smoothing_factor), color="red")
    plt.title("Mean Tardiness")
    plt.show()

    plt.plot(episode_list, earliness_list, alpha=0.2, color="blue")
    plt.plot(episode_list, smoothing(earliness_list, alpha=smoothing_factor), color="blue")
    plt.title("Mean Earliness")
    plt.show()

    plt.plot(episode_list, on_time_list, alpha=0.2, color="black")
    plt.plot(episode_list, smoothing(on_time_list, alpha=smoothing_factor), color="black")
    plt.ylim([0, 1])
    plt.title("On-Time Ratio")
    plt.show()

