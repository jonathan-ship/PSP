import json, os, math, random
import pandas as pd
import numpy as np
from collections import OrderedDict

if __name__ == "__main__":
    sheet_name_list = ["SAW1", "SAW2", "SAW3"]

    output_data = OrderedDict()
    due_date_data = dict()

    initial_date = pd.to_datetime('2022-09-27', format='%Y-%m-%d')
    for sheet in sheet_name_list:
        output_data[sheet] = OrderedDict()
        data = pd.read_excel("../data/actual data.xlsx", sheet_name=sheet)
        data = data.drop([0, 1])
        data.reset_index()
        column_list = ["Date", "호선", "블록", "부재번호", "P", "S", "web_width", "web_thickness", "face_width",
                       "face_thickness", "location", "temperature", "weld_size", "length"]
        data = data.rename(columns={data.columns[i]: column_list[i] for i in range(14)})

        data["P"].replace(math.nan, 0, inplace=True)
        data["S"].replace(math.nan, 0, inplace=True)

        date = "NaT"
        series = "nan"
        block = "nan"
        steel = "nan"
        web_width, web_thickness, face_width, face_thickness, weld_size, length = math.nan, math.nan, math.nan, math.nan, math.nan, math.nan

        for idx in range(len(data)):
            temp = data.iloc[idx]
            date = temp["Date"] if str(temp["Date"]) != "NaT" else date

            # SAW1 ~ SAW3에 대한 데이터가 모두 존재하는 2022년 9월 27일 ~ 2022년 11월 08일만 선택
            if (date >= pd.to_datetime('2022-09-27', format='%Y-%m-%d')) and (
                    date <= pd.to_datetime('2022-11-08', format='%Y-%m-%d')):
                # 날짜를 정수형으로 (현재 날짜 - 시뮬레이션 시작 날짜)
                date_int = (date - initial_date).days
                if date_int not in output_data[sheet].keys():
                    output_data[sheet][date_int] = list()

                # 엑셀 필드가 비어있다면 이전 값과 동일한 값으로, 아니라면 입력된 값
                series = str(temp["호선"]) if str(temp["호선"]) != "nan" else series
                block = str(temp["블록"]) if str(temp["블록"]) != "nan" else block
                steel = str(temp["부재번호"]) if str(temp["부재번호"]) != "nan" else steel
                num_steel = int(temp["P"] + temp["S"])
                web_width = temp["web_width"] if not math.isnan(temp["web_width"]) else web_width
                web_thickness = temp["web_thickness"] if not math.isnan(temp["web_thickness"]) else web_thickness
                face_width = temp["face_width"] if not math.isnan(temp["face_width"]) else face_width
                face_thickness = temp["face_thickness"] if not math.isnan(temp["face_thickness"]) else face_thickness
                if type(temp["weld_size"]) == str:
                    weld_size = 7.0
                elif temp["weld_size"] == 65.0:  # 정규각장 중 값 하나가 65mm로 되어 있어 6.5mm로 수정
                    weld_size = 6.5
                else:
                    weld_size = temp["weld_size"] if not math.isnan(temp["weld_size"]) else weld_size
                length = temp["length"] if not math.isnan(temp["length"]) else length

                if series != "nan":
                    block_name = str(series) + "_" + str(block)
                    if block_name not in due_date_data.keys():
                        due_date_data[block_name] = random.randint(0, 44)
                    block_data = {"name": "{0}_{1}".format(block_name, steel), "num_steel": int(num_steel),
                                  "web_width": float(web_width), "web_thickness": float(web_thickness),
                                  "face_width": float(face_width), "face_thickness": float(face_thickness),
                                  "weld_size": float(weld_size), "length": float(length)}
                    output_data[sheet][date_int].append(list(block_data.values()))

    with open('actual_data.json', 'w') as f:
        json.dump(output_data, f)
    with open('due_date.json', 'w') as f:
        json.dump(due_date_data, f)
