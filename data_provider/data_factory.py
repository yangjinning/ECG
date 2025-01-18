from data_provider.data_loader import Dataset_ECG_Single_InbatchCL
from torch.utils.data import DataLoader
import os

def data_provider(args, flag, num_data=None):

    root_path_ecg = args.root_path_ecg
    root_path_json = args.root_path_json
    shuffle_flag = args.shuffle_flag
    scale = args.scale #是否做归一化
    batch_size = args.batch_size
    drop_last = True

    path_json = os.path.join(root_path_json,f"template_{flag}.json")

    data_set = Dataset_ECG_Single_InbatchCL(
        root_path_ecg=root_path_ecg,
        path_json=path_json,
        flag=flag,  # train,valid,test?
        shuffle_flag=shuffle_flag,
        scale=scale,
        num_data=num_data
    )
    if flag == "test":
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            num_workers=10,
            drop_last=drop_last)
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            num_workers=10,
            drop_last=drop_last,
            collate_fn=data_set.collator)
    return data_set, data_loader




