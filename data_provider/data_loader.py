import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
import json
import random
import wfdb
import torch
import pickle as pkl
from tqdm import tqdm

warnings.filterwarnings('ignore')

class Dataset_ECG_Single_InbatchCL(Dataset):
    def __init__(self, root_path_ecg, path_json,flag,shuffle_flag,scale,num_data=None):
        self.root_path_ecg = root_path_ecg
        assert flag in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.shuffle_flag = shuffle_flag

        if self.shuffle_flag:
            random.seed(42)

        self.lookup_table = None
        self.json_data = None

        if self.set_type != 2:
            if os.path.exists("./dataset/mimic-iv-ecg/pkl/json_data.pkl"):
                with open("./dataset/mimic-iv-ecg/pkl/json_data.pkl", "rb") as f:
                    self.json_data = pkl.load(f)

            if os.path.exists("./dataset/mimic-iv-ecg/pkl/lookup_table.pkl"):
                with open("./dataset/mimic-iv-ecg/pkl/lookup_table.pkl", "rb") as f:
                    self.lookup_table = pkl.load(f)

            if (self.lookup_table is None) or (self.json_data is None):
                self.lookup_table, self.json_data = self.__lookup_table__(path_json)

            if num_data:
                self.json_data = self.json_data[:num_data]
        else:
            if num_data:
                self.json_data = self.__load_json__(path_json)[:num_data]
            else:
                self.json_data = self.__load_json__(path_json)

        self.scale = scale
        if self.scale:
            self.scaler = StandardScaler()  # 使其均值为 0，标准差为 1

    def __load_json__(self,path_json):
        with open(path_json, "r") as f:
            records = json.load(f)
        if self.shuffle_flag:
            random.shuffle(records)
        return records

    def __lookup_table__(self, path_json):
        with open(path_json, "r") as f:
            data = json.load(f)

        lookup_table = {}
        json_data = []
        for item in tqdm(data):
            question = item["question"]
            ecg_id = item["ecg_id"][0]  # 取出唯一的 ecg_id

            # 初始化question_id对应的字典
            if question not in lookup_table:
                lookup_table[question] = {"ecgid":{},"data":[]}


            lookup_table[question]["ecgid"][ecg_id] = item["answer"]
            lookup_table[question]["data"].append(item)


        q_list = list(lookup_table.keys())
        if self.shuffle_flag: random.shuffle(q_list)
        for q in q_list:
            data = lookup_table[q]["data"][:]
            if self.shuffle_flag: random.shuffle(data)
            json_data.extend(data)

        with open("./dataset/mimic-iv-ecg/pkl/json_data.pkl","wb") as f:
            pkl.dump(json_data, f)
            print("Save json_data.pkl")

        with open("./dataset/mimic-iv-ecg/pkl/lookup_table.pkl","wb") as f:
            pkl.dump(lookup_table, f)
            print("Save lookup_table.pkl")

        return lookup_table,json_data

    def collator(self, samples): #将一组样本组合成一个批次后返回
        if len(samples) == 0:
            return {}

        batch_samples = list(zip(*samples))

        if self.set_type == 2:
            return batch_samples

        batch_question = []
        batch_ecgid = []
        batch_ans = []
        batch_score = []
        for i in samples:
            info = eval(i[-1])
            batch_question.append(info["question"])
            batch_ecgid.append(info["ecg_id"][0])
            batch_ans.append(info["answer"])

        for idx0,question in enumerate(batch_question):
            gt = batch_ans[idx0]
            scores = []
            for idx1, ecgid in enumerate(batch_ecgid):
                score = 0
                if idx0 == idx1:
                    score = 1
                if ecgid in self.lookup_table[question]["ecgid"]:
                    ans = self.lookup_table[question]["ecgid"][ecgid]
                    score = len(set(gt) & set(ans)) * 2 / (len(gt) + len(ans))
                scores.append(score)
            batch_score.append(scores)
        batch_samples.append(batch_score)
        batch_samples[0] = torch.tensor(batch_samples[0])
        batch_samples[-1] = torch.tensor(batch_samples[-1])
        return batch_samples

    def __load_ecg_data__(self, root_path_ecg, filepath):
        data = wfdb.rdsamp(os.path.join(root_path_ecg, filepath))
        signal, meta = data
        return np.array(signal)

    def __getitem__(self, index):
        example = self.json_data[index]
        example_str = json.dumps(example)
        ecg_data = self.__load_ecg_data__(self.root_path_ecg, example["ecg_path"])
        if self.scale:
            self.scaler.fit(ecg_data) #归一化没有问题，均值逼近0，标准差为1
            ecg_data = self.scaler.transform(ecg_data)
        else:
            ecg_data = ecg_data

        question = example["question"]
        answer_text = ".".join(example["answer"])
        return ecg_data,question,answer_text,example_str

    def __len__(self):
        return  len(self.json_data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
