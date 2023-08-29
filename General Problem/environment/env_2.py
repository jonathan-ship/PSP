import os
import numpy as np

from environment.simulation import *


class PMSP:
    def __init__(self, num_job=200, num_m=5, episode=1, test_sample=None, reward_weight=None, rule_weight=None,
                 ddt=None, pt_var=None, is_train=True):
        self.num_job = num_job  # scheduling 대상 job의 수
        self.num_m = num_m  # parallel machine 수
        self.episode = episode
        self.test_sample = test_sample
        self.reward_weight = reward_weight if reward_weight is not None else [0.5, 0.5]
        self.rule_weight = rule_weight
        self.ddt = ddt
        self.pt_var = pt_var
        self.is_train = is_train

        self.state_dim = 14 + num_m * 8
        self.action_dim = 4

        self.mapping = {0: "SSPT", 1: "ATCS", 2: "MDD", 3: "COVERT"}
        self.time = 0
        self.previous_time_step = 0

        self.time_list = list()
        self.tardiness_list = list()
        self.setup_list = list()

        self.done = False
        self.time = 0
        self.reward_setup = 0
        self.reward_tard = 0

        self.sim_env, self.model, self.source, self.sink, self.routing, self.monitor = self._modeling()

    def step(self, action):
        done = False
        self.previous_time_step = self.sim_env.now
        routing_rule = self.mapping[action]

        self.routing.decision.succeed(routing_rule)
        self.routing.indicator = False

        while True:
            if self.routing.indicator:
                if self.sim_env.now != self.time:
                    self.time = self.sim_env.now
                break

            if self.sink.completed == self.num_job:
                done = True
                self.sim_env.run()
                # if self.episode % 50 == 0:
                #     self.monitor.save_tracer()
                # # self.monitor.save_tracer()
                break

            if len(self.sim_env._queue) == 0:
                self.monitor.get_logs(file_path='log.csv')
            self.sim_env.step()

        reward = self._calculate_reward()
        next_state = self._get_state()

        return next_state, reward, done

    def _modeling(self):
        env = simpy.Environment()

        monitor = Monitor()
        monitor.reset()

        iat = 15 / self.num_m

        model = dict()
        job_list = list()
        for j in range(self.num_job):
            job_name = "Job {0}".format(j)
            processing_time = np.random.uniform(10, 20) if self.test_sample is None else self.test_sample['processing_time'][j]
            feature = random.randint(0, 5) if self.test_sample is None else self.test_sample['feature'][j]
            job_list.append(Job(name=job_name, processing_time=processing_time, feature=feature))

        routing = Routing(env, model, monitor, end_num=self.num_job, weight=self.rule_weight)
        routing.reset()

        sink = Sink(env, monitor)
        sink.reset()

        source = Source(env, job_list, iat, self.ddt, routing, monitor)

        for m in range(self.num_m):
            machine_name = "Machine {0}".format(m)
            model[machine_name] = Process(env, machine_name, routing, sink, monitor, pt_var=self.pt_var)
            model[machine_name].reset()

        return env, model, source, sink, routing, monitor

    def reset(self):
        self.episode = self.episode + 1 if self.episode > 1 else 1  # episode
        if self.is_train:
            self.pt_var = np.random.uniform(low=0.1, high=0.5)
            self.ddt = np.random.uniform(low=0.8, high=1.2)

        self.sim_env, self.model, self.source, self.sink, self.routing, self.monitor = self._modeling()
        self.done = False
        self.reward_setup = 0
        self.reward_tard = 0
        self.monitor.reset()

        while True:
            # Check whether there is any decision time step
            if self.routing.indicator:
                break

            self.sim_env.step()

        return self._get_state()

    def _get_state(self):
        f_1 = np.zeros(4)  # tardiness level of jobs in rouitng.queue
        f_2 = np.zeros(3)  # tightness level min, avg, max
        f_3 = np.zeros(2)  # routing을 요청한 경우 해당 machine의 셋업값, queue에 있는 job 중 해당 셋업값과 같은 셋업값을 갖는 job의 수
        f_4 = np.zeros(self.num_m)  # 각 machine의 셋업 상태
        f_5 = np.zeros(
            [self.num_m, 6])  # machine 별 각 셋업값을 가지는 job의 비율(routing.queue에 있는 job들 중에서) / 가능한 셋업 경우의 수 : 0~5, 6가지
        f_6 = np.zeros(1)  # completion rate
        f_7 = np.zeros(self.num_m)  # 각 machine에서의 progress rate
        f_8 = np.zeros(2)  # tardiness level index (x, v)
        f_9 = np.zeros(2)  # setup index (x, v)

        input_queue = copy.deepcopy(list(self.routing.queue.items))
        tt_list = list()
        if len(input_queue) > 0:
            g_1 = 0
            g_2 = 0
            g_3 = 0
            g_4 = 0

            for job in input_queue:
                job_dd = job.due_date
                max_tightness = job_dd - (1 + self.pt_var) * job.processing_time - self.sim_env.now
                min_tightness = job_dd - (1 - self.pt_var) * job.processing_time - self.sim_env.now
                tt_list.append(job_dd - job.processing_time - self.sim_env.now)

                if max_tightness > 0:
                    g_1 += 1
                elif max_tightness <= 0 and min_tightness > 0:
                    g_2 += 1
                elif min_tightness <= 0 and self.sim_env.now > job_dd:
                    g_3 += 1
                elif self.sim_env.now < job_dd:
                    g_4 += 1
                else:
                    print(0)

            f_1[0] = g_1 / len(input_queue)
            f_1[1] = g_2 / len(input_queue)
            f_1[2] = g_3 / len(input_queue)
            f_1[3] = g_4 / len(input_queue)

        f_2[0] = np.min(tt_list) if len(tt_list) > 0 else 0.0
        f_2[1] = np.mean(tt_list) if len(tt_list) > 0 else 0.0
        f_2[2] = np.max(tt_list) if len(tt_list) > 0 else 0.0

        calling_line = self.model[self.routing.machine]
        setting = calling_line.setup
        f_3[0] = setting / 5

        same_feature = 0
        for job in input_queue:
            if job.feature == setting:
                same_feature += 1
        f_3[1] = same_feature / len(input_queue) if len(input_queue) > 0 else 0.0

        for i in range(self.num_m):
            line = self.model["Machine {0}".format(i)]
            line_setup = line.setup
            f_4[i] = line_setup / 5

            for job in input_queue:
                setup_time = abs(job.feature - line_setup)
                f_5[i, setup_time] += 1 / len(input_queue)

            f_7[i] = (self.sim_env.now - line.start_time) / (line.expected_finish_time - line.start_time) if line.job is not None else 0.0

        f_6[0] = self.sink.completed / self.num_job

        if self.sim_env.now > 0:
            self.time_list.append(self.sim_env.now)
            setup_ratio = self.monitor.setup / self.routing.created if self.routing.created else 0.0
            self.setup_list.append(setup_ratio)
            self.tardiness_list.append(self.monitor.tardiness / self.sim_env.now)

        f_8[0] = self.setup_list[-1] if len(self.setup_list) else 0.0
        f_9[0] = self.tardiness_list[-1] if len(self.tardiness_list) else 0.0
        if len(self.time_list) > 1:
            v_setup = (self.setup_list[-1] - self.setup_list[-2]) / (self.time_list[-1] - self.time_list[-2])
            f_8[1] = 1 / (1 + np.exp(-v_setup))
            v_tard = (self.tardiness_list[-1] - self.tardiness_list[-2]) / (
                    self.time_list[-1] - self.time_list[-2])
            f_9[1] = 1 / (1 + np.exp(-v_tard))

        state = np.concatenate((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9), axis=None)
        return state

    def _calculate_reward(self):
        reward_1 = - self.routing.setup / 5
        self.reward_setup -= self.routing.setup / 5

        reward_2 = 0.0
        if len(self.sink.tardiness) > 0:
            for tardiness in self.sink.tardiness:
                reward_2 += np.exp(-tardiness) - 1

        # reward_2 = np.exp(-self.sink.tardiness) - 1
        self.reward_tard += reward_2

        reward = reward_1 * self.reward_weight[1] + reward_2 * self.reward_weight[0]
        self.routing.setup = 0
        self.sink.tardiness = list()
        return reward

    def get_logs(self, path=None):
        log = self.monitor.get_logs(path)
        return log

