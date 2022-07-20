import math, json
import pandas as pd
import numpy as np


if __name__ == "__main__":
    sheet_name_list = ["SAW1", "SAW2", "SAW3"]

    output_data = dict()
    for sheet in sheet_name_list:
        data = pd.read_excel("./data/actual data.xlsx", sheet_name=sheet)
        data = data.drop([0, 1])
        data.reset_index()
        column_list = ["Date", "호선", "블록", "부재번호", "P", "S", "web_width", "web_thickness", "face_width",
                       "face_thickness", "location", "temperature", "weld_size", "length"]
        data = data.rename(columns={data.columns[i]: column_list[i] for i in range(14)})

        data["P"].replace(math.nan, 0, inplace=True)
        data["S"].replace(math.nan, 0, inplace=True)

        series = "nan"
        block = "nan"
        steel = "nan"
        web_width, web_thickness, face_width, face_thickness, weld_size, length = math.nan, math.nan, math.nan, math.nan, math.nan, math.nan

        for idx in range(len(data)):
            temp = data.iloc[idx]
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
            elif temp["weld_size"] == 65.0:
                weld_size = 6.5
            else:
                weld_size = temp["weld_size"] if not math.isnan(temp["weld_size"]) else weld_size
            length = temp["length"] if not math.isnan(temp["length"]) else length

            if series != "nan":
                block_name = str(series) + "_" + str(block)
                if block_name not in output_data.keys():
                    output_data[block_name] = dict()
                if "{0}_{1}".format(block_name, steel) not in output_data[block_name].keys():
                    output_data[block_name]["{0}_{1}".format(block_name, steel)] = {"num_steel": int(num_steel),
                                                                                    "web_width": float(web_width),
                                                                                    "web_thickness": float(web_thickness),
                                                                                    "face_width": float(face_width),
                                                                                    "face_thickness": float(face_thickness),
                                                                                    "weld_size": float(weld_size),
                                                                                    "length": float(length)}
                else:
                    output_data[block_name]["{0}_{1}".format(block_name, steel)]["num_steel"] += num_steel

    with open('block_sample.json', 'w') as f:
        json.dump(output_data, f)
