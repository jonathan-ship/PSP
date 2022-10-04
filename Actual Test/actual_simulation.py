import simpy, math, copy
import pandas as pd
import numpy as np

from datetime import timedelta

class Steel:
    def __init__(self, name, block, steel, feature, due_date, processed_date):
        # 해당 job의 이름
        self.name = name
        self.block = block
        self.steel = steel
        self.length = feature["length"]
        self.due_date = due_date
        self.planned_processed_date = processed_date

        self.web_face = [feature["web_width"], feature["web_thickness"], feature["face_width"],
                         feature["face_thickness"]]

        self.avg_speed = 1200 - ((feature["weld_size"] - 4.5) / 0.5) * 50
        self.avg_pt = feature["length"] / self.avg_speed

        self.completed = 0  # 종료 시간


class Process:
    def __init__(self, env, name, model, monitor):
        self.env = env
        self.name = name
        self.model = model
        self.monitor = monitor

        self.queue = simpy.Store(env)
        self.idle = True
        self.job = None
        self.block = None
        self.day = None
        self.day_finished_time = dict()

        self.total_setup = 0

        env.process(self.run())

    def run(self):
        while len(self.queue.items) > 0:
            setup_time = 0
            setup_memo = None
            part = yield self.queue.get()

            if self.day is None:
                self.day = part.planned_processed_date
                self.day_finished_time[self.day] = [self.env.now]
            elif self.day != part.planned_processed_date:
                self.day_finished_time[self.day].append(self.env.now)
                self.day_finished_time[part.planned_processed_date] = [self.env.now]
                self.day = part.planned_processed_date

            if (self.job is not None) and (part.web_face != self.job.web_face):
                setup_time = 5
                setup_memo = "{0} to {1}".format(self.job.web_face, part.web_face)
                self.monitor.setup += 1
                self.monitor.setup_list.append(1)
                self.total_setup += 1
            elif (self.job is not None) and (part.web_face == self.job.web_face):
                self.monitor.record(part=part.name, block=part.block, event="Non-SetUp",
                                    process=self.name, memo="{0} to {1}".format(self.job.web_face, part.web_face))

            self.block = part.block
            self.job = part
            self.idle = False

            # 실제 작업 시간
            processing_time = np.random.uniform(low=part.avg_pt * 0.9, high=part.avg_pt * 1.1)

            # 다음 날로 이동
            # if (self.env.now + setup_time + part.avg_pt) % 1440 > 960:
            #     self.monitor.record(time=self.env.now, event="Day Off")
            #     today = math.floor(self.env.now / 1440)
            #     next_day = today + 1 if today % 7 != 5 else today + 2
            #     to_next_day = next_day * 1440 - self.env.now
            #     yield self.env.timeout(to_next_day)
            #     self.monitor.record(time=self.env.now, event="Day On")

            # 셋업 발생
            if setup_memo is not None:
                self.monitor.record(part=part.name, block=part.block, event="Set-Up",
                                    process=self.name, memo=setup_memo)
                yield self.env.timeout(setup_time)

            # 셋업 후 작업
            self.monitor.record(day=self.day, part=part.name, block=part.block, event="Work Start",
                                process=self.name, memo=processing_time)

            yield self.env.timeout(processing_time)  # 0.9 * 평균 ~ 1.1 * 평균 사이의 stochastic한 값
            self.monitor.record(part=part.name, block=part.block, event="Work Finish",
                                process=self.name)

            self.model["Sink"].put(part)
            self.idle = True

        self.day_finished_time[self.day].append(self.env.now)

    def reset(self):
        self.queue.items = list()
        self.idle = True
        self.job = None
        self.block = None
        self.setup = 0  # 0 - No Setup / 1 - Setup


class Sink:
    def __init__(self, env, block_info, monitor):
        self.env = env
        self.block_info = block_info
        self.monitor = monitor

        # 블록별 작업이 종료된 Job의 수
        self.finished = dict()
        self.finished_block = list()
        self.total_finish = 0

    def put(self, job):
        if job.block not in self.finished.keys():
            self.finished[job.block] = dict()
            self.finished[job.block]["num"] = 0
            self.finished[job.block]["time"] = [self.env.now]

        self.finished[job.block]["num"] += 1  # jobtype 별 종료 개수
        self.total_finish += 1

        self.monitor.record(part=job.name, block=job.block, event="Completed", process="Sink")

        if self.finished[job.block]["num"] == self.block_info[job.block]["num_steel"]:
            self.finished_block.append([job.block, job.planned_processed_date])  # State 및 Reward 시 사용
            difference = int(job.planned_processed_date - self.block_info[job.block]["due_date"])  # due date 대비 얼마나 늦게 끝났는 지 (현재 시각 - 납기일)
            self.monitor.record(part=job.name, block=job.block, event="Block Completed",
                                process="Sink", memo=difference)

            if difference > 0:  # tardiness
                self.monitor.tardiness.append(difference)

            self.finished[job.block]["time"].append(self.env.now)

    def reset(self):
        self.finished = dict()
        self.finished_block = list()
        self.total_finish = 0


class Monitor:
    def __init__(self, filepath):
        self.day = list()
        self.part = list()
        self.block = list()
        self.event = list()
        self.process = list()
        self.memo = list()

        self.filepath = filepath

        self.setup = 0
        self.setup_list = list()
        self.tardiness = list()

    def record(self, day=None, part=None, block=None, event=None, process=None, memo=None):
        if day is not None:
            actual_date = (pd.to_datetime("2022-09-27", format="%Y-%m-%d") + timedelta(days=day)).date()
        else:
            actual_date = None
        self.day.append(actual_date)
        self.part.append(part)
        self.block.append(block)
        self.event.append(event)
        self.process.append(process)
        self.memo.append(memo)

    def save_tracer(self):
        event_tracer = pd.DataFrame(columns=["Day", "Part", "Block", "Event", "Process", "Memo"])
        event_tracer["Day"] = self.day
        event_tracer["Part"] = self.part
        event_tracer["Block"] = self.block
        event_tracer["Event"] = self.event
        event_tracer["Process"] = self.process
        event_tracer["Memo"] = self.memo
        event_tracer.to_csv(self.filepath, encoding='utf-8-sig')
        # print(self.filepath)

    def reset(self):
        self.day = list()
        self.part = list()
        self.block = list()
        self.event = list()
        self.process = list()
        self.memo = list()

        self.setup = 0
        self.tardiness = list()

