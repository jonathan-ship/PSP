import json
import numpy as np

from environment.simulation import *


class WeldingLine:
    def __init__(self, num_block=240, num_line=3, start_lag=0, log_dir=None, reward_weight=None):
        self.num_block = num_block
        self.num_line = num_line
        self.start_lag = start_lag
        self.log_dir = log_dir
        self.reward_weight = reward_weight if reward_weight is not None else [1, 1, 1]

        with open('../block_sample.json', 'r') as f:
            self.block_sample = json.load(f)

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

        while True:
            if self.routing.indicator:
                if self.sim_env.now != self.time:
                    self.time = self.sim_env.now
                break

            if self.num_jobs == self.model["Sink"].total_finish:
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
        self.sim_block = dict()
        self.done = False

        self.sim_env, self.model, self.routing, self.monitor = self._modeling()

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
        week3_due_date = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
        due_date_list = np.random.choice(week3_due_date, size=self.num_block)
        # due_date_list = list(np.random.randint(low=0, high=6, size=self.num_block))
        block_list = np.random.choice([key for key in self.block_sample.keys()], size=self.num_block)

        # simulation object modeling
        model = dict()
        env = simpy.Environment()
        monitor = Monitor(self.log_dir + '/log_{0}.csv'.format(self.e))
        monitor.reset()

        model["Source"] = Source(env)
        self.sim_block = dict()

        self.num_jobs = 0

        for block_idx in range(len(block_list)):
            block_name = block_list[block_idx]
            block_due_date = due_date_list[block_idx]

            self.sim_block["Block_{0}".format(block_idx + 1)] = dict()
            self.sim_block["Block_{0}".format(block_idx + 1)]["due_date"] = block_due_date
            self.sim_block["Block_{0}".format(block_idx + 1)]["num_steel"] = 0

            steel_idx = 1
            for steel_name in self.block_sample[block_name].keys():
                for i in range(self.block_sample[block_name][steel_name]["num_steel"]):
                    self.sim_block["Block_{0}".format(block_idx + 1)][
                        "Steel_{0}_{1}_{2}".format(block_idx + 1, steel_idx, i + 1)] = Steel(
                        name="Steel_{0}_{1}_{2}".format(block_idx + 1, steel_idx, i + 1),
                        block="Block_{0}".format(block_idx + 1), steel="Steel_{0}_{1}".format(block_idx + 1, steel_idx),
                        feature=self.block_sample[block_name][steel_name], due_date=block_due_date)
                    model["Source"].queue.put(copy.deepcopy(self.sim_block["Block_{0}".format(block_idx + 1)][
                                                                "Steel_{0}_{1}_{2}".format(block_idx + 1, steel_idx,
                                                                                           i + 1)]))

                    self.sim_block["Block_{0}".format(block_idx + 1)]["num_steel"] += 1
                    self.num_jobs += 1
                steel_idx += 1

        routing = Routing(env, model, self.sim_block, monitor)

        for i in range(self.num_line):
            model["Line {0}".format(i + 1)] = Process(env, "Line {0}".format(i + 1), model, routing, monitor)
            model["Line {0}".format(i + 1)].reset()

        model["Sink"] = Sink(env, self.sim_block, monitor)
        model["Sink"].reset()

        return env, model, routing, monitor

    def _get_state(self):
        # define 4 features
        f_1 = np.zeros(self.num_line)  # Setup -> 현재 라인의 셋업값과 같은 셋업인 부재의 수
        f_2 = np.zeros(4)  # Due Date -> Tardiness Level for non-setup
        f_3 = np.zeros(4)  # Due Date -> Tardiness Level for setup
        f_4 = np.zeros(self.num_line)  # General Info -> 각 라인에서의 남은 작업시간

        input_queue = copy.deepcopy(self.model["Source"].queue.items)

        # Feature 1, 4
        for line_num in range(self.num_line):
            line = self.model["Line {0}".format(line_num + 1)]
            if line.job is not None:
                line_feature = line.job.web_face
                same_setup_list = [1 for job in input_queue if job.web_face == line_feature]

                f_1[line_num] = np.sum(same_setup_list) / len(input_queue) if len(input_queue) > 0 else 0.0
                f_4[line_num] = (line.planned_finish_time - self.sim_env.now) / line.planned_finish_time if not line.idle else 0
            else:
                f_1[line_num] = 1.0

        # Feature 2, 3
        calling_line = self.model[self.routing.line]
        if calling_line.job is not None:
            setting = calling_line.job.web_face

            non_setup_list = list()
            setup_list = list()

            if len(input_queue) > 0:
                for job in input_queue:
                    if job.web_face == setting:
                        non_setup_list.append(job)
                    else:
                        setup_list.append(job)

            def _cal_expected_finish_time(avg_pt, num_jobs):
                expected_time = self.sim_env.now
                for _ in range(num_jobs):
                    if (expected_time + avg_pt) % 1440 <= 960:
                        expected_time += avg_pt
                    else:
                        day = math.floor(expected_time / 1440)
                        next_day = day + 1 if day % 7 != 5 else day + 2
                        # next_day = day + 1
                        expected_time = next_day * 1440 + avg_pt

                return expected_time

            # Feature 2
            if len(non_setup_list) > 0:
                g_1 = 0
                g_2 = 0
                g_3 = 0
                g_4 = 0

                for non_setup_job in non_setup_list:
                    job_dd = non_setup_job.due_date * 1440 + 960
                    finished_jobs = self.model["Sink"].finished[non_setup_job.block]["num"] if non_setup_job.block in \
                                                                                        self.model[
                                                                                            "Sink"].finished.keys() else 0

                    num_residual = self.sim_block[non_setup_job.block]["num_steel"] - finished_jobs

                    max_tightness = job_dd - _cal_expected_finish_time(non_setup_job.avg_pt * 1.1, num_residual)
                    min_tightness = job_dd - _cal_expected_finish_time(non_setup_job.avg_pt * 0.9, num_residual)

                    if max_tightness > 0:
                        g_1 += 1
                    elif (max_tightness <= 0) and (min_tightness > 0):
                        g_2 += 1
                    elif (min_tightness <= 0) and (self.sim_env.now > job_dd):
                        g_3 += 1
                    elif self.sim_env.now < job_dd:
                        g_4 += 1
                    else:
                        print(0)

                f_2[0] = g_1 / len(non_setup_list)
                f_2[1] = g_2 / len(non_setup_list)
                f_2[2] = g_3 / len(non_setup_list)
                f_2[3] = g_4 / len(non_setup_list)

            # Feature 3
            if len(setup_list) > 0:
                g_1 = 0
                g_2 = 0
                g_3 = 0
                g_4 = 0

                for setup_job in setup_list:
                    job_dd = setup_job.due_date * 1440 + 960
                    finished_jobs = self.model["Sink"].finished[setup_job.block]["num"] if setup_job.block in \
                                                                                        self.model[
                                                                                            "Sink"].finished.keys() else 0

                    num_residual = self.sim_block[setup_job.block]["num_steel"] - finished_jobs

                    max_tightness = job_dd - _cal_expected_finish_time(setup_job.avg_pt * 1.1, num_residual)
                    min_tightness = job_dd - _cal_expected_finish_time(setup_job.avg_pt * 0.9, num_residual)

                    if max_tightness > 0:
                        g_1 += 1
                    elif (max_tightness <= 0) and (min_tightness > 0):
                        g_2 += 1
                    elif (min_tightness <= 0) and (self.sim_env.now > job_dd):
                        g_3 += 1
                    elif self.sim_env.now < job_dd:
                        g_4 += 1
                    else:
                        print(0)

                f_3[0] = g_1 / len(setup_list)
                f_3[1] = g_2 / len(setup_list)
                f_3[2] = g_3 / len(setup_list)
                f_3[3] = g_4 / len(setup_list)

        state = np.concatenate((f_1, f_2, f_3, f_4), axis=None)
        return state

    def _calculate_reward(self):
        self.reward = 0
        # setup
        self.reward -= self.reward_weight[0] * self.monitor.setup * 0.1

        # Earliness / Tardiness
        for block_info in self.model["Sink"].finished_block:
            block_name = block_info[0]
            completed_time = block_info[1]
            difference_time = self.sim_block[block_name]["due_date"] - completed_time

            if difference_time < 0:  # tardiness
                self.reward += self.reward_weight[2] * (np.exp(difference_time) - 1)
            # else:  # earliness
            #     self.reward += self.reward_weight[1] * (np.exp(-difference_time) - 1)

        self.monitor.setup = 0
        self.model["Sink"].finished_block = list()
        return self.reward


if __name__ == "__main__":
    welding_line = WeldingLine()
    welding_line._modeling()