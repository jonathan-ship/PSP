import simpy, copy, json, math
import scipy.stats as stats
import pandas as pd
import numpy as np
from collections import OrderedDict
from environment.simulation import *


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


class WeldingLine:
    def __init__(self, num_block=80, num_line=3, start_lag=0, log_dir=None, reward_weight=None):
        self.num_block = num_block
        self.num_line = num_line
        self.start_lag = start_lag
        self.thickness_type, self.thickness_prob, self.steel_dist, self.length_list = read_data()
        self.log_dir = log_dir
        self.reward_weight = reward_weight if reward_weight is not None else [1, 1, 1]

        self.done = False
        self.tardiness = 0.0
        self.e = 1
        self.time = 0
        self.setup = 0
        self.num_jobs = 0

        self.total_setup = 0
        self.tardiness = 0
        self.earliness = 0

        self.sim_env, self.model, self.routing, self.monitor = self._modeling()

    def step(self, action):
        done = False
        self.previous_time_step = self.sim_env.now

        self.routing.decision.succeed(action)
        self.routing.indicator = False
        if (self.model[self.routing.line].block is None) or (self.model[self.routing.line].block != action + 1):
            self.setup = 1
            self.total_setup += 1

        while True:
            if self.routing.indicator:
                if self.sim_env.now != self.time:
                    self.time = self.sim_env.now
                break

            if self.num_jobs == self.model["Sink"].num_finished_job:
                done = True
                self.sim_env.run()
                if self.e % 50 == 0:
                    self.monitor.save_tracer()
                # self.monitor.save_tracer()
                break
            if len(self.sim_env._queue) == 0:
                self.monitor.save_tracer()
            self.sim_env.step()

        reward = self._calculate_reward()
        next_state = self._get_state()

        return next_state, reward, done

    def reset(self):
        self.e += 1  # episode
        self.total_setup = 0
        self.setup = 0
        self.num_steel_list = list()

        self.sim_env, self.model, self.routing, self.monitor = self._modeling()
        self.done = False
        self.monitor.reset()
        for name in self.model.keys():
            if name != "Source":
                self.model[name].reset()
        self.routing.reset()

        self.tardiness = 0
        self.earliness = 0

        while True:
            # Check whether there is any decision time step
            if self.routing.indicator:
                break
            self.sim_env.step()

        return self._get_state()

    def _modeling(self):
        # data modeling
        self.num_steel_list = [math.ceil(eval(self.steel_dist)) for _ in range(self.num_block)]
        self.num_jobs = np.sum(self.num_steel_list)
        self.due_date_list = list(np.random.randint(low=self.start_lag, high=self.start_lag + 6, size=self.num_block))
        self.lengths = list(np.random.choice(self.length_list, size=self.num_block))
        thickness_list = list(np.random.choice(self.thickness_type, self.num_block, p=self.thickness_prob))

        # simulation object modeling
        model = dict()
        env = simpy.Environment()
        monitor = Monitor(self.log_dir + '/result_{0}.csv'.format(self.e))

        model["Source"] = Source(env)

        for idx in range(len(self.num_steel_list)):
            for j in range(self.num_steel_list[idx]):
                model["Source"].queue.put(
                    Steel("{0}_{1}".format(idx + 1, j + 1), idx + 1, self.lengths[idx], thickness_list[idx],
                          self.due_date_list[idx]))

        routing = Routing(env, model, monitor)

        for i in range(self.num_line):
            model["Line {0}".format(i + 1)] = Process(env, "Line {0}".format(i + 1), model, routing, monitor)

        model["Sink"] = Sink(env, self.num_block, monitor)

        return env, model, routing, monitor

    def _get_state(self):
        # define 7 features
        f_1 = np.zeros(self.num_block)  # Input Queue에서 각 라인에서 작업 중인 부재와 같은 블록인 부재의 수
        f_2 = np.zeros(4)  # Tardiness Level
        f_3 = np.zeros(self.num_line)  # 각 라인의 셋업 값
        f_4 = np.zeros(self.num_line)  # 각 라인에서의 남은 작업시간

        input_queue = copy.deepcopy(self.model["Source"].queue.items)

        # Feature 1
        for i in range(self.num_block):
            block = i + 1
            same_block = [1 for job in input_queue if job.block == block]

            f_1[i] = np.sum(same_block) / self.num_steel_list[i]

        # Feature 2
        g_1 = 0
        g_2 = 0
        g_3 = 0
        g_4 = 0

        if len(input_queue) > 0:
            for i in range(self.num_block):
                same_block_list = [non_processed for non_processed in input_queue if non_processed.block == (i + 1)]
                if len(same_block_list) > 0:
                    block_dd = (self.due_date_list[i] + 1) * 1440
                    block_pt = self.lengths[i] / same_block_list[0].avg_speed
                    tightness = block_dd - self.sim_env.now

                    if tightness > block_pt * 1.1:
                        g_1 += 1
                    elif (tightness > block_pt * 0.9) and (tightness <= block_pt * 1.1):
                        g_2 += 1
                    elif (tightness > 0) and (tightness <= block_pt * 0.9):
                        g_3 += 1
                    elif tightness <= 0:
                        g_4 += 1
                else:  # input queue에 해당 블록이 없으면 -> 이미 전부 작업
                    g_1 += 1

            f_2[0] = g_1 / self.num_block
            f_2[1] = g_2 / self.num_block
            f_2[2] = g_3 / self.num_block
            f_2[3] = g_4 / self.num_block

        # Feature 3, 4
        for line_num in range(self.num_line):
            line = self.model["Line {0}".format(line_num + 1)]
            if line.block is not None:
                f_3[line_num] = line.block / self.num_block if line.block is not None else 0
                f_4[line_num] = (line.planned_finish_time - self.sim_env.now) / line.planned_working_time if not line.idle else 0

        state = np.concatenate((f_1, f_2, f_3, f_4), axis=None)
        return state

    def _calculate_reward(self):
        self.reward = 0
        # setup
        self.reward -= self.reward_weight[0] * self.setup
        self.monitor.record(time=self.sim_env.now, event="Setup Reward", memo=-self.reward_weight[0] * self.setup)

        # Earliness / Tardiness
        for job in self.model["Sink"].finished_job:
            difference_time = job.due_date - job.completed
            if difference_time < 0:  # tardiness
                self.reward += self.reward_weight[2] * (np.exp(difference_time) - 1)
            else:  # earliness
                self.reward += self.reward_weight[1] * (np.exp(-difference_time) - 1)

        self.setup = 0
        self.model["Sink"].finished_job = list()
        return self.reward


if __name__ == "__main__":
    welding_line = WeldingLine()
    welding_line._modeling()