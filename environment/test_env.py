import json, os
import numpy as np

from test_simulation import *


if __name__ == "__main__":
    num_block = 240
    num_line = 3
    reward_weight = [1, 1, 1]

    # on_time_list = list()
    tardiness_list = list()
    # earliness_list = list()
    setup_list = list()
    in_queue_list = list()
    rule = "RANDOM"

    with open('../block_sample.json', 'r') as f:
        block_sample = json.load(f)

    log_path = './test_sim/3Weeks/{0}'.format(rule)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    for episode in range(1):
        print("- Episode {0}".format(episode))
        # due_date_list = list(np.random.randint(low=0, high=6, size=num_block))
        week3_due_date = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
        due_date_list = np.random.choice(week3_due_date, size=num_block)
        block_list = np.random.choice([key for key in block_sample.keys()], size=num_block)

        # simulation object modeling
        model = dict()
        env = simpy.Environment()
        monitor = Monitor(log_path + '/log_dir_{0}.csv'.format(episode))
        monitor.reset()

        model["Source"] = Source(env)
        sim_block = dict()

        for block_idx in range(len(block_list)):
            block_name = block_list[block_idx]
            block_due_date = due_date_list[block_idx]

            sim_block["Block_{0}".format(block_idx + 1)] = dict()
            sim_block["Block_{0}".format(block_idx + 1)]["due_date"] = block_due_date
            sim_block["Block_{0}".format(block_idx + 1)]["num_steel"] = 0

            steel_idx = 1
            for steel_name in block_sample[block_name].keys():
                for i in range(block_sample[block_name][steel_name]["num_steel"]):
                    sim_block["Block_{0}".format(block_idx + 1)][
                        "Steel_{0}_{1}_{2}".format(block_idx + 1, steel_idx, i + 1)] = Steel(
                        name="Steel_{0}_{1}_{2}".format(block_idx + 1, steel_idx, i + 1),
                        block="Block_{0}".format(block_idx + 1), steel="Steel_{0}_{1}".format(block_idx + 1, steel_idx),
                        feature=block_sample[block_name][steel_name], due_date=block_due_date)
                    model["Source"].queue.put(copy.deepcopy(sim_block["Block_{0}".format(block_idx + 1)][
                                                                "Steel_{0}_{1}_{2}".format(block_idx + 1, steel_idx,
                                                                                           i + 1)]))

                    sim_block["Block_{0}".format(block_idx + 1)]["num_steel"] += 1
                steel_idx += 1

        routing = Routing(env, model, sim_block, monitor, routing=rule)

        for i in range(num_line):
            model["Line {0}".format(i + 1)] = Process(env, "Line {0}".format(i + 1), model, routing, monitor)
            model["Line {0}".format(i + 1)].reset()

        model["Sink"] = Sink(env, sim_block, monitor)
        model["Sink"].reset()

        env.run()
        # monitor.save_tracer()

        # on_time_list.append(monitor.on_time / num_block)
        tardiness_list.append(sum(monitor.tardiness) / num_block)
        # earliness_list.append(sum(monitor.earliness) / num_block)
        setup_list.append(sum(monitor.setup_list) / model["Sink"].total_finish)

        total_in_queue = 0.0
        for block in model["Sink"].finished.keys():
            total_in_queue += (model["Sink"].finished[block]["time"][1] - model["Sink"].finished[block]["time"][
                0]) / 1440

        in_queue_list.append(total_in_queue / num_block)

    # print("On Time Ratio  : {0}".format(round(np.mean(on_time_list), 2)))
    print("Avg. Tardiness : {0}".format(round(np.mean(tardiness_list), 2)))
    # print("Avg. Earliness : {0}".format(round(np.mean(earliness_list), 2)))
    print("Setup Ratio    : {0}".format(round(np.mean(setup_list), 2)))
    print("Avg. In_queue Time : {0}".format(np.mean(in_queue_list)))



