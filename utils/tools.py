import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from IPython.display import clear_output
from tqdm import tqdm
import json

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.7 ** ((epoch - 1) // 1))}  # 每个epoch学习率减半
        # lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))} #每个epoch学习率减半
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=5, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
                self.accelerator.print('\tsave checkpoint')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
                self.accelerator.print('\tsave checkpoint')
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth') #lora,加llm_basemodel参数，全都存下来了
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_loader):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_ecg_data, batch_question, batch_answer, _, batch_score) in tqdm(enumerate(vali_loader)):
            batch_ecg_data = batch_ecg_data.float().to(accelerator.device)
            batch_score = batch_score.float().to(accelerator.device)
            loss = model(batch_ecg_data, batch_question, batch_answer, batch_score)
            total_loss.append(loss.detach().cpu().numpy())
    total_loss = np.average(total_loss)
    model.train()
    return total_loss


def test(args, accelerator, model, test_loader):
    results = []
    model.eval()
    with torch.no_grad():
        for i, (batch_ecg_data, batch_question, batch_answer, example_str) in tqdm(enumerate(test_loader)):

            batch_wo_answer = tuple("" for i in batch_answer)  # 要把答案去掉
            batch_ecg_data = batch_ecg_data.float().to(accelerator.device)  # (32,96,1)

            outputs = model(batch_ecg_data, batch_question, batch_wo_answer)

            llm_model, pad_prompt_embs_batch, pad_attention_mask_batch, pad_labels_batch,tokenizer = outputs

            B,num_input_ids,_ = pad_prompt_embs_batch.shape

            generate_ids = llm_model.generate(inputs_embeds=pad_prompt_embs_batch[:, :-1,:],
                                            attention_mask=pad_attention_mask_batch[:, :-1],
                                            do_sample=False,
                                            num_beams=1,
                                            max_new_tokens=600)


            prediction_list = tokenizer.batch_decode(generate_ids,
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False
                                                )

            results.extend(post_proc(example_str, prediction_list))

    cal_acc_qtype(results)
    model.train()
    return results


def post_proc_cls(example_str, prediction_list,batch_labels):
    results = []
    for idx,item in enumerate(example_str):
        item = json.loads(f"[{item}]")[0]
        item["prediction"] = prediction_list[idx]
        item["label"] = batch_labels[idx]
        results.append(item)
    return results

def cal_acc_qtype_cls(j):
    correct = {'single-verify': 0,"single-query": 0,  "single-choose": 0, "comparison_irrelevant-verify": 0, "comparison_consecutive-verify": 0}
    total = {'single-verify': 0,"single-query": 0,  "single-choose": 0, "comparison_irrelevant-verify": 0, "comparison_consecutive-verify": 0}
    for i in j:
        if set(i["prediction"]) == set(i["label"]):
            correct[i["question_type"]] += 1
        total[i["question_type"]] += 1
    print("Correct Stat:", correct)
    print("Total Stat:", total)
    print("Acc Ratio:")
    for k in total:
        if total[k] !=0:
            print(k,":{:.2f}".format(correct[k]/total[k] * 100))


def post_proc(example_str, prediction_list):
    results = []
    for item, prediction in zip(example_str, prediction_list):
        ans = prediction.strip()
        item = json.loads(f"[{item}]")[0]
        item["prediction"] = ans.split(".")
        results.append(item)
    return results

def cal_acc_qtype(j):
    correct = {'single-verify': 0,"single-query": 0,  "single-choose": 0, "comparison_irrelevant-verify": 0, "comparison_consecutive-verify": 0}
    total = {'single-verify': 0,"single-query": 0,  "single-choose": 0, "comparison_irrelevant-verify": 0, "comparison_consecutive-verify": 0}
    for i in j:
        if set(i["prediction"]) == set(i["answer"]):
            correct[i["question_type"]] += 1
        total[i["question_type"]] += 1
    print("Correct Stat:", correct)
    print("Total Stat:", total)
    print("Acc Ratio:")
    for k in total:
        if total[k] !=0:
            print(k,":{:.2f}".format(correct[k]/total[k] * 100))

def cal_acc_atype(j):
    correct = {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0}
    total = {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0}
    for i in j:
        if set(i["prediction"]) == set(i["answer"]):
            correct[i["attribute_type"]] += 1
        total[i["attribute_type"]] += 1
    print(correct)
    print(total)
    for k in total:
        print(k,":{:.2f}".format(correct[k]/total[k] * 100))

def cal_detail_acc(j):
    correct = {'single-verify': {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0},
               'single-query': {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0},
               "single-choose": {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0}}
    total = {'single-verify': {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0},
               'single-query': {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0},
               "single-choose": {"scp_code":0,"numeric_feature":0,"heart_axis":0,"stage_of_infarction":0,"noise":0}}
    for i in j:
        if set(i["prediction"]) == set(i["answer"]):
            correct[i["question_type"]][i["attribute_type"]] += 1
        total[i["question_type"]][i["attribute_type"]] += 1
    print("=" * 50)
    print(correct)
    print(total)
    for k1 in total:
        print("="*50)
        print(k1)
        for k2 in total[k1]:
            if total[k1][k2] != 0:
                print(k2,":{:.2f}".format(correct[k1][k2]/total[k1][k2] * 100))
            else:
                print("None")



def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content

def plot_losses(train_steps, train_losses, val_steps, val_losses, path):
    clear_output(wait=True)  # 清除之前的图像
    plt.figure(figsize=(10, 5))
    # 绘制训练损失
    plt.plot(train_steps, train_losses, label='Train Loss', marker='o')
    # 绘制验证损失
    plt.plot(val_steps, val_losses, label='Validation Loss', marker='x')
    # 添加标签和图例
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(path + "/loss.png", format='png', dpi=300)