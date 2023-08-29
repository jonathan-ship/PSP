import json, copy, os
import numpy as np


def get_sample(num_block, tt_number):
    with open("SHI_sample.json", 'r') as f:
        data = json.load(f)

    test = dict()

    for i in range(tt_number):
        test["test_{0}".format(i + 1)] = dict()
        test["test_{0}".format(i + 1)]["block_list"] = list(np.random.choice([key for key in data.keys()], size=num_block))
        if num_block == 240:
            week3_due_date = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19]
            due_date_list = np.random.choice(week3_due_date, size=num_block)
        elif num_block == 160:
            week2_due_date = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            due_date_list = np.random.choice(week2_due_date, size=num_block)
        else:
            due_date_list = list(np.random.randint(low=0, high=6, size=num_block))  # Week 1 버전
        # test_240["test_{0}".format(i + 1)] = list(np.random.choice([key for key in new_data.keys()], size=240))
        test["test_{0}".format(i+1)]["due_date"] = [int(temp) for temp in due_date_list]

    return test