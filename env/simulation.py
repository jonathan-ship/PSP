import simpy, math, copy
import pandas as pd
import numpy as np


class Steel:
    def __init__(self, name, block, length, thickness, due_date):
        # 해당 job의 이름
        self.name = name
        self.block = block
        self.length = length
        self.thickness = thickness
        self.due_date = due_date

        self.avg_speed = 1200 - ((thickness - 4.5) / 0.5) * 50
        self.completed = 0


class Source:
    def __init__(self, env):
        self.env = env
        self.name = "Source"

        self.queue = simpy.FilterStore(env)


class Process:
    def __init__(self, env, name, model, routing, monitor):
        self.env = env
        self.name = name
        self.model = model
        self.routing = routing
        self.monitor = monitor

        self.queue = simpy.Store(env)
        self.idle = True
        self.job = None
        self.block = None
        self.setup = 0  # 0 - No Setup / 1 - Setup
        self.planned_finish_time = 0
        self.planned_working_time = 0

        env.process(self.run())

    def run(self):
        self.monitor.record(time=self.env.now, event="Request Routing for Job", process=self.name,
                            memo=len(self.model["Source"].queue.items))
        yield self.env.process(self.routing.run(location=self.name))

        while True:
            setup_time = 0
            setup_memo = None
            part = yield self.queue.get()
            self.idle = False

            if (self.job is not None) and (part.block != self.job.block):
                setup_time = 5
                setup_memo = "{0} to {1}".format(self.job.block, part.block)
                self.setup = 1
                self.monitor.setup += 1

            self.job = part
            self.block = part.block

            # State 계산을 위한 예상 작업 시간 - 평균값 사용
            self.planned_working_time = part.length / part.avg_speed
            self.planned_finish_time = self.env.now + self.planned_working_time
            # 실제 작업 시간
            processing_time = part.length / np.random.uniform(low=part.avg_speed * 0.9, high=part.avg_speed * 1.1)

            # 다음 날로 이동
            if (self.env.now + setup_time + (part.length / part.avg_speed)) % 1440 > 960:
                self.monitor.record(time=self.env.now, event="Day Off")
                to_next_day = (math.floor(self.env.now/1440) + 1) * 1440 - self.env.now
                yield self.env.timeout(to_next_day)
                self.monitor.record(time=self.env.now, event="Day On")

            # 셋업 발생
            if setup_memo is not None:
                self.monitor.record(time=self.env.now, part=part.name, block=part.block, event="Set-Up", process=self.name,
                                    memo=setup_memo)
                yield self.env.timeout(setup_time)

            # 셋업 후 작업
            self.monitor.record(time=self.env.now, part=part.name, block=part.block, event="Work Start",
                                process=self.name, memo=processing_time)

            yield self.env.timeout(processing_time)  # 0.9 * 평균 ~ 1.1 * 평균 사이의 stochastic한 값
            self.monitor.record(time=self.env.now, part=part.name, block=part.block, event="Work Finish",
                                process=self.name)

            self.model["Sink"].put(part)
            self.idle = True

            if len(self.model["Source"].queue.items) > 0:
                self.monitor.record(time=self.env.now, event="Request Routing for Job", process=self.name,
                                    memo=len(self.model["Source"].queue.items))
                yield self.env.process(self.routing.run(location=self.name))
            elif (len(self.queue.items) == 0) and (len(self.model["Source"].queue.items) == 0):
                break

    def reset(self):
        self.idle = True
        self.job = None
        self.block = None
        self.setup = 0  # 0 - No Setup / 1 - Setup
        self.planned_finish_time = 0


class Routing:
    def __init__(self, env, model, monitor):
        self.env = env
        self.model = model
        self.monitor = monitor

        self.waiting = env.event()

        self.indicator = False
        self.decision = False
        self.line = None

        self.idle = False
        self.job = None

        self.waiting_jobs = copy.deepcopy([job.block for job in self.model["Source"].queue.items])

    def run(self, location=None):
        if len(self.model["Source"].queue.items) > 0:
            self.indicator = True

            self.decision = self.env.event()
            self.line = location
            block = yield self.decision
            block += 1
            self.monitor.record(time=self.env.now, block=block, event="Put Action In Routing class", process=location)
            self.decision = None
            #
            # blocks = [job.block for job in self.model["Source"].queue.items]

            next_job = yield self.model["Source"].queue.get(lambda x: x.block == block)

            self.monitor.record(time=self.env.now, part=next_job.name, block=next_job.block, event="Routing Finished",
                                process=location)

            self.model[location].queue.put(next_job)

    def reset(self):
        self.indicator = False
        self.decision = False
        self.line = None

        self.idle = False
        self.job = None

        self.waiting_jobs = copy.deepcopy([job.block for job in self.model["Source"].queue.items])


class Sink:
    def __init__(self, env, num_block, monitor):
        self.env = env
        self.num_block = num_block
        self.monitor = monitor

        # 블록별 작업이 종료된 Job의 수
        # self.finished = {idx: 0 for idx in range(num_block)}
        self.num_finished_job = 0
        self.finished_job = list()

    def put(self, job):
        self.finished_job.append(job)
        # self.finished[job.block] += 1  # jobtype 별 종료 개수
        self.num_finished_job += 1  # 전체 종료 개수
        job.completed = math.floor(self.env.now/1440)
        day = math.floor(self.env.now / 1440)
        difference = int(day - job.due_date)
        self.monitor.record(time=self.env.now, part=job.name, block=job.block, event="Completed", process="Sink",
                            memo=difference)

        if difference > 0:  # tardiness
            self.monitor.tardiness.append(difference)
        elif difference < 0:  # earliness
            self.monitor.earliness.append(-difference)
        elif difference == 0:
            self.monitor.on_time += 1

    def reset(self):
        self.finished = {idx: 0 for idx in range(self.num_block)}
        self.num_finished_job = 0
        self.finished_job = list()


class Monitor:
    def __init__(self, filepath):
        self.day = list()
        self.time = list()
        self.part = list()
        self.block = list()
        self.event = list()
        self.process = list()
        self.memo = list()

        self.filepath = filepath

        self.setup = 0
        self.tardiness = list()
        self.earliness = list()
        self.on_time = 0

    def record(self, time=None, part=None, block=None, event=None, process=None, memo=None):
        self.time.append(round(time, 2))
        self.day.append(int(math.floor(time/1440)))
        self.part.append(part)
        self.block.append(block)
        self.event.append(event)
        self.process.append(process)
        self.memo.append(memo)

    def save_tracer(self):
        event_tracer = pd.DataFrame(columns=["Day", "Part", "Block", "Event", "Process", "Memo", "Time"])
        event_tracer["Day"] = self.day
        event_tracer["Part"] = self.part
        event_tracer["Block"] = self.block
        event_tracer["Event"] = self.event
        event_tracer["Process"] = self.process
        event_tracer["Memo"] = self.memo
        event_tracer["Time"] = self.time
        event_tracer.to_csv(self.filepath, encoding='utf-8-sig')
        print(self.filepath)

    def reset(self):
        self.day = list()
        self.time = list()
        self.part = list()
        self.block = list()
        self.event = list()
        self.process = list()
        self.memo = list()

        self.setup = 0
        self.tardiness = list()
        self.earliness = list()
        self.on_time = 0

