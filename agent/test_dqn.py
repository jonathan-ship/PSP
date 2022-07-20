import os
import pandas as pd

from dqn import *
from environment.env import WeldingLine


if __name__ == "__main__":
    state_size = 14
    action_size = 4

    log_path = '../result/model/dqn'

    event_path = '../test/result/dqn'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    # on_time_list = list()
    tardiness_list = list()
    # earliness_list = list()
    setup_list = list()
    in_queue_list = list()

    env = WeldingLine(log_dir=event_path)
    q = Qnet(state_size, action_size)
    q.load_state_dict(torch.load(log_path + '/episode4600.pt')["model_state_dict"])

    for i in range(100):
        print("Episode {0}".format(i + 1))
        env.e = i
        step = 0
        done = False
        state = env.reset()
        r = list()

        while not done:
            epsilon = 0
            step += 1
            action = q.sample_action(torch.from_numpy(state).float(), epsilon)

            # 환경과 연결
            next_state, reward, done = env.step(action)
            r.append(reward)
            state = next_state

            if done:
                env.monitor.save_tracer()
                break

        # on_time_list.append(env.monitor.on_time / env.num_block)
        tardiness_list.append(sum(env.monitor.tardiness) / env.num_block)
        # earliness_list.append(sum(env.monitor.earliness) / env.num_block)
        setup_list.append(sum(env.monitor.setup_list) / env.model["Sink"].total_finish)
        total_in_queue = 0.0
        for block in env.model["Sink"].finished.keys():
            total_in_queue += (env.model["Sink"].finished[block]["time"][1] - env.model["Sink"].finished[block]["time"][
                0]) / 1440

        in_queue_list.append(total_in_queue / env.num_block)

    # print("On Time Ratio  : {0}".format(round(np.mean(on_time_list), 2)))
    print("Avg. Tardiness : {0}".format(round(np.mean(tardiness_list), 2)))
    # print("Avg. Earliness : {0}".format(round(np.mean(earliness_list), 2)))
    print("Setup Ratio    : {0}".format(round(np.mean(setup_list), 2)))
    print("Avg. In_queue Time : {0}".format(np.mean(in_queue_list)))