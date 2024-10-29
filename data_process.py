import time
import numpy as np
import wfdb #波形数据库软件包（WFDB）是用于读取、写入以及处理WFDB信号和注释的一组工具库，其核心功能围绕着心电信号等生理数据的分析展开
import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy.fftpack import fft, ifft 
from scipy import signal
from datasets import Dataset

"""PTB-XL REPORT DATASET"""
# PTB-XL csv ： /data/PyProjects/ECG-data/PTB-XL/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv
path_ptb = "/data/PyProjects/ECG-data/PTB-XL/physionet.org/files/ptb-xl/1.0.3/"
csv_ptb = pd.read_csv(path_ptb +'ptbxl_database.csv', index_col='ecg_id')
# 保留字符数量大于1的report
# translate to english
# save in json



"""MIMIC-IV-ECG REPORT DATASET"""
# Mimic-IV-ECG csv ： /data/PyProjects/ECG-data/MIMIC-ECG/machine_measurements.csv
path_mimic = "/data/PyProjects/ECG-data/MIMIC-ECG/"
csv_mimic = pd.read_csv(path_mimic+'machine_measurements.csv', index_col='ecg_id')
# concat report_0 - report_17 using comma
# 保留字符数量大于1的report

"""PTB-XL QA DATASET"""
# 只保留single类型的问题
# 根据ecg_id匹配心电图文件
# 分批加载数据集

"""MIMIC-IV-ECG QA DATASET"""
# 只保留single类型的问题
# 根据ecg_id匹配心电图文件
# 分批加载数据集


def load_dataset(path, sampling_rate, release=False):
    # load and convert annotation data
    Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path)

    return X, Y

def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data

def compute_label_aggregations(df, folder, ctype):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0] #只有疾病诊断类别
        if ctype == 'diagnostic': #添加scp_code对应的在scp_statements.csv中的index既scp_code
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic': #添加scp_code对应的diagnostic_subclass
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic': #添加scp_code对应的diagnostic_class  --用这个
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df

def select_data(XX,YY, ctype, min_samples):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter 
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter 
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    return X, Y, y, mlb





    X, y = preprocess(X, y)

    # --- Combine and split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2, shuffle = True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 2, shuffle = True)

    # --- Get embeddings

    train_embeddings, train_input_ids, train_words = get_embeddings(y_train[:, 0], bert_model, tokenizer, device)
    val_embeddings, val_input_ids, val_words  = get_embeddings(y_val[:, 0], bert_model, tokenizer, device)
    test_embeddings, test_input_ids, test_words  = get_embeddings(y_test[:, 0],bert_model ,tokenizer, device)

    # -- Dataset
    # 只保留要训练的主函数接收的参数，其他删掉，不然会报错
    # LLM接收：input_ids,labels,attention_mask
    train_dataset = ECGDataset(
        report_labels = train_embeddings,
        disease_labels = y_train[:, 1].astype('int32'),
        signals = X_train,
        input_ids = train_input_ids,
        words = train_words
    )
    val_dataset = ECGDataset(
        report_labels = val_embeddings,
        disease_labels = y_val[:, 1].astype('int32'),
        signals = X_val,
        input_ids = val_input_ids,
        words = val_words
    )

    test_dataset = ECGDataset(
        report_labels = test_embeddings,
        disease_labels = y_test[:, 1].astype('int32'),
        signals = X_test,
        input_ids = test_input_ids,
        words = test_words
    )

    # --- Loader
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle = True)

    valid_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            shuffle=True)