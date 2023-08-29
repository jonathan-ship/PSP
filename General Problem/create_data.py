import json, os
from numpyencoder import NumpyEncoder
import numpy as np
import random


def get_sample(num_job, test_num):
    sample_dict = dict()

    for i in range(test_num):
        sample_dict[i] = dict()
        sample_dict[i]['processing_time'] = list(np.random.uniform(10, 20, size=num_job))
        sample_dict[i]['feature'] = list(np.random.randint(0, 6, size=num_job))

    return sample_dict



if __name__ == "__main__":
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for num_job in [600, 800]:
        sample_dict = dict()

        for i in range(500):
            sample_dict[i] = dict()
            sample_dict[i]['processing_time'] = list(np.random.uniform(10, 20, size=num_job))
            sample_dict[i]['feature'] = list(np.random.randint(0, 6, size=num_job))

        with open(data_dir + '/sample{0}.json'.format(num_job), 'w') as f:
            json.dump(sample_dict, f, cls=NumpyEncoder)