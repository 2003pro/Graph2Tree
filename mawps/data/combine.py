# -*- coding: utf-8 -*-
import json
from tqdm import tqdm
import os

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def write_json(path,file):
    with open(path,'w') as f:
        json.dump(file,f)
        
def load_raw_data(filename):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % 7 == 0:  # every 7 line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data

def combine_json(graph_dict, ori_json):
    new_data = []
    for i in tqdm(range(len(ori_json))):
        item = ori_json[i]
        item['group_num'] = graph_dict[item['id']]['group_num']
        new_data.append(item)        
    print('Graph has been inserted into ori_json!')
    return new_data

def main(whole_path, save_path, ori_path):
    whole = read_json(whole_path)
    whole_dict = dict([(item['id'],item) for item in whole])
    ori = load_raw_data(ori_path)
    new_data = combine_json(whole_dict, ori)
    write_json(save_path, new_data)
    print('Finished')
    
whole_path = './whole_processed.json'
ori_path = './Math_23K.json'
save_path = './Math_23K_processed.json'
main(whole_path, save_path, ori_path)