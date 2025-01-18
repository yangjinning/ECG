import shutil
import os
import json

for type in ["train","test","valid"]:
    with open(f"/data/PyProjects/ECG_CL/dataset/mimic-iv-ecg/combine/template_{type}.json","r")as f:
        j = json.load(f)
        for i in j:
            path = "/".join(i["ecg_path"].split("/")[:-1])
            src_path = "/data/PyProjects/ECG-data/MIMIC-ECG/files/" + path
            dst_path = "/data/PyProjects/ECG_CL/dataset/mimic-iv-ecg/data/" + path
            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path)


# src_path = "/data/PyProjects/ECG-data/MIMIC-ECG/files/p1279/p12799029/s45102411/"
# dst_path = "/data/PyProjects/ECG_CL/dataset/mimic-iv-ecg/data/p1279/p12799029/s45102411/"
# shutil.copytree(src_path, dst_path)