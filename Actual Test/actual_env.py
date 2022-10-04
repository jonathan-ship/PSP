import os

import simpy, json
from actual_simulation import *


if __name__ == "__main__":
    # 블록 정보
    with open('actual_data.json', 'r') as f:
        data = json.load(f)

    # 납기일 정보
    with open('due_date.json', 'r') as f:
        due_date_data = json.load(f)

    # 이벤트 트레이서
    log_path = './result'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Modeling
    env = simpy.Environment()
    monitor = Monitor(log_path + '/event.csv')
    monitor.reset()

    block_info = dict()
    model = dict()
    for saw in data.keys():
        model["Line {0}".format(int(saw[-1]))] = Process(env, "Line {0}".format(int(saw[-1])), model, monitor)

        for processing_date in data[saw].keys():
            steel_list = data[saw][processing_date]
            for each_steel in steel_list:
                steel_name = each_steel[0]
                temp_name = steel_name.split('_')
                block_name = "{0}_{1}".format(temp_name[0], temp_name[1])
                feature = {"web_width": each_steel[2], "web_thickness": each_steel[3], "face_width": each_steel[4],
                           "face_thickness": each_steel[5], "weld_size": each_steel[6], "length": each_steel[7]}
                if block_name not in block_info.keys():
                    block_info[block_name] = dict()
                    block_info[block_name] = {"due_date": due_date_data[block_name], "num_steel": 0}

                for idx in range(each_steel[1]):
                    model["Line {0}".format(int(saw[-1]))].queue.put(Steel("{0}_{1}".format(steel_name, idx),
                                                                           block_name, steel_name, feature,
                                                                           due_date_data[block_name],
                                                                           int(processing_date)))
                    block_info[block_name]["num_steel"] += 1


    model["Sink"]= Sink(env, block_info, monitor)
    model["Sink"].reset()

    env.run()
    monitor.save_tracer()

    line_result = {"Line 1": model["Line 1"].day_finished_time, "Line 2": model["Line 2"].day_finished_time,
                   "Line 3": model["Line 3"].day_finished_time}

    result_output = dict()
    for line_number in line_result.keys():
        result = line_result[line_number]
        for day in result.keys():
            actual_date = (pd.to_datetime("2022-09-27", format="%Y-%m-%d") + timedelta(days=day)).date()
            if actual_date not in result_output.keys():
                result_output[actual_date] = {"Line 1": None, "Line 2": None, "Line 3": None}
            result_output[actual_date][line_number] = result[day][1] - result[day][0]
    df_result_output = pd.DataFrame.from_dict(result_output)
    df_result_output = df_result_output.transpose()
    df_result_output.to_excel("actual_result.xlsx")

    print("Line 1 Setup : ", model["Line 1"].total_setup)
    print("Line 2 Setup : ", model["Line 2"].total_setup)
    print("Line 3 Setup : ", model["Line 3"].total_setup)
