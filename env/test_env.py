import json, math, os
import scipy.stats as stats
from test_simulation import *
import matplotlib.pyplot as plt


def read_data():
    with open('../pre-processing.json', 'r') as f:
        data = json.load(f)

    # thickness
    thickness_type = list()
    thickness_prob = list()
    for thickness, prob in data['thickness'].items():
        thickness_type.append(float(thickness))
        thickness_prob.append(prob)

    # materials per block
    info = data["Material per block"]
    steel_dist = "stats.{0}.rvs({1}, {2}, loc={3}, scale={4})".format(info['type'], info['a'], info['b'], info['loc_beta'], info['scale_beta'])

    # length
    length_list = data["length"]

    return thickness_type, thickness_prob, steel_dist, length_list


if __name__ == "__main__":
    num_block = 80
    num_line = 3
    reward_weight = [1, 1, 1]

    thickness_type, thickness_prob, steel_dist, length_list = read_data()

    setup_list = list()
    tardiness_list = list()
    earliness_list = list()

    reward_list = list()

    for episode in range(50):
        num_steel_list = [math.ceil(eval(steel_dist)) for _ in range(num_block)]
        num_jobs = np.sum(num_steel_list)
        due_date_list = list(np.random.randint(low=0, high=5, size=num_block))
        lengths = list(np.random.choice(length_list, size=num_block))
        thickness_list = list(np.random.choice(thickness_type, num_block, p=thickness_prob))

        model = dict()
        env = simpy.Environment()

        log_dir = "./test_sim"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        monitor = Monitor(log_dir + "/test_{0}.csv".format(episode))
        monitor.reset()

        model["Source"] = Source(env)

        for idx in range(len(num_steel_list)):
            for j in range(num_steel_list[idx]):
                model["Source"].queue.put(
                    Steel("{0}_{1}".format(idx, j + 1), idx, lengths[idx], thickness_list[idx],
                          due_date_list[idx]))

        routing = Routing(env, model, monitor)

        for i in range(num_line):
            model["Line {0}".format(i + 1)] = Process(env, "Line {0}".format(i + 1), model, routing, monitor)
            model["Line {0}".format(i + 1)].reset()

        model["Sink"] = Sink(env, num_block, monitor)
        env.run()
        print(num_jobs)
        monitor.save_tracer()

        tardiness = np.sum(monitor.tardiness)
        earliness = np.sum(monitor.earliness)
        setup = np.sum(monitor.setup)
        reward = tardiness + earliness + setup

        tardiness_list.append(tardiness)
        earliness_list.append(earliness)
        setup_list.append(setup)
        reward_list.append(reward)

        print("episode {0} : tardiness = {1}   earliness = {2}   setup = {3}   reward = {4}".format(episode + 1,
                                                                                                    tardiness,
                                                                                                    earliness, setup,
                                                                                                    reward))

    print("Avg.Tardiness: ", np.mean(tardiness_list))
    print("Avg.Earliness: ", np.mean(earliness_list))
    print("Avg.Set-Up: ", np.mean(setup_list))
    print("Avg.Reward: ", np.mean(reward_list))