from math import sqrt
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer,BitsAndBytesConfig, AutoModelForCausalLM,AutoTokenizer
from layers.Embed import PatchEmbedding
import transformers
from transformers.models.llama import LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
from datetime import datetime

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2) #展平从倒数第二个维度开始，将patch平铺
        self.linear = nn.Linear(nf, target_window) #将展平后的张量的每个元素映射到 target_window 维度的空间。nf 是输入的特征数，而 target_window 是输出的目标预测长度
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x) #(32,1,32,12) -> (32,1,384)
        x = self.linear(x) #(32,1,96)
        x = self.dropout(x)
        return x


class TimeLLMInBatchCL(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(TimeLLMInBatchCL, self).__init__()

        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            configs.local_llm_path,
            # 'huggyllama/llama-7b',
            # trust_remote_code=True,
            local_files_only=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(configs.local_llm_path, local_files_only=True)
        self.tokenizer.padding_side = "left"

        if not self.tokenizer.pad_token:
            pad_token = '[PAD]'
            num_new_tokens = self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.llm_model.get_input_embeddings().weight.data
                output_embeddings = self.llm_model.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

        self.word_embeddings = self.llm_model.get_input_embeddings().weight

        for param in self.llm_model.parameters():
            param.requires_grad = False


        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.n_vars, self.patch_len, self.stride, configs.dropout) # 16, 16, 8, 0.1

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # d_model:16, n_heads, d_keys = None, d_llm:768 llm hidden emb维度
        # d_model:16, n_heads = 8, d_keys = 128, d_llm:4096 llm hidden emb维度
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        #如果 stride=2，表示每次提取一个 patch 后，窗口会向右移动 2 个时间步，从而跳过中间的 2 个时间步。
        # 原本计算公式：floor((configs.seq_len - self.patch_len) / self.stride) + 1;
        # 先把最后一个patch拿出来，所以先减掉patch_len；然后每stride个取一次，才能确保每次都会有一个patch，且不会超出整个的范围；再把取出来的patch加回去
        # 这里+2, 是因为Embed.py的reprogramming_laye的ReplicationPad1d，会在最后pad上一个stride长度的矩阵（复制最后一个emb），使得stride最后不会舍掉一部分
        # 实际：floor((configs.seq_len - self.patch_len + self.stride) / self.stride) + 1;  既等于上面的 +2
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2) # （96 - 16） / 8 + 2
        # stride = patch_len / 2; patch_nums <= 336; seq_len = 5000; 求 patch_len 33-34
        # patch_len=36; stride=18; seq_len=5000; patch_nums=277
        # patch_len=32; stride=16; seq_len=5000; patch_nums=312
        #TODO: patch_len = d_model

        self.head_nf = self.d_ff * self.patch_nums # 32 * 12； d_ff：dimension of fcn 32


        if configs.use_lora:
            lora_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.05,
                r=64,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "embed_tokens",
                                "lm_head"],
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "lm_head"],
                # layers_to_transform=[31],
            )

            self.llm_model = get_peft_model(self.llm_model, lora_config)
            self.llm_model.print_trainable_parameters()

    def forward(self, batch_ecg_data, batch_question, batch_answer, batch_score=None):
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        # 对文本emb进行线性变换
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1,0)  # (1000,768); 将llm词表压缩到1000个 self.word_embeddings(30522,768) 转置成(768，30522) 线性变换成(768，1000)再转置 (1000，768)

        # x_enc时序编码
        x_enc = batch_ecg_data.permute(0, 2, 1).contiguous()  # (32,1,96) #(5,5000,12)->(5,12,5000)

        # 初始化都是float32，但是训练时用的bf16，因为deepspeed config里设置 bf16 enabled:True
        enc_out, x_reduced, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        # enc_out, x_reduced, n_vars = self.patch_embedding(x_enc)  # (32,1,96)只有1个变量->(32,12,16),1 将时序编码patch（切分）成12份，每隔8个取一个patch，每个patch长16
        # enc_out (5,12,5000) -> (5,312,32); x_reduced (5,1,5000)

        B, N, T = x_reduced.size()  # （32，96，1）#(5,1,5000) N代表 channel 个数 = n_vars
        x_reduced =  x_reduced.reshape(B * N, T, 1).to(torch.float32) #（32，96，1）batch*变量数量，时间序列长度，1 #(5,1,5000) -> (5,5000,1)
        #(5,5000,12) -> ()
        min_values = torch.min(x_reduced, dim=1)[0] #（32，1） 在每个时序（单个sample的单变量）上求 ecg(60,1)
        max_values = torch.max(x_reduced, dim=1)[0] #（32，1）
        medians = torch.median(x_reduced, dim=1).values #（32，1）
        lags = self.calcute_lags(x_reduced)
        trends = x_reduced.diff(dim=1).sum(dim=1)

        prompt_prefix = []
        prompt_suffix = []
        ans = []
        for batch_idx in range(B):
            min_values_str = "{:.4f}".format(min_values[batch_idx].tolist()[0])
            max_values_str = "{:.4f}".format(max_values[batch_idx].tolist()[0])
            median_values_str = "{:.4f}".format(medians[batch_idx].tolist()[0])
            lags_values_str = str(lags[batch_idx].tolist())
            question = batch_question[batch_idx]
            ans_ = batch_answer[batch_idx]

            #TODO:是否改掉这些统计特征，更换特殊字符区别TS和text
            prompt_prefix_ = (
                f"<|start_prompt|>Dataset description: The dataset contains electrocardiogram (ECG) time-series data. "
                f"Task description: Given a patient's ECG signal, answer a clinical question related to cardiac health based on the ECG data. "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of the signal is {'upward' if trends[batch_idx] > 0 else 'downward'}, "
                f"top 5 lags are: {lags_values_str}. "
                f"Question:{question}"
                "<|end_prompt|>"
                "<|start_ecg|>"
            )
            prompt_suffix_ = ("<|end_ecg|>Answer:")

            prompt_prefix.append(prompt_prefix_)
            prompt_suffix.append(prompt_suffix_)
            ans.append(ans_ + self.tokenizer.eos_token)

        prompt_prefix = self.tokenizer(prompt_prefix, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids # 32,136 / 32,135# 如果批次中最长序列的长度超过 max_length=2048：在这种情况下，序列会被截断
        prompt_suffix = self.tokenizer(prompt_suffix, return_tensors="pt", padding=True, truncation=True,max_length=512).input_ids  # 32,136 / 32,135# 如果批次中最长序列的长度超过 max_length=2048：在这种情况下，序列会被截断
        ans = self.tokenizer(ans, return_tensors="pt", padding=True, truncation=True ,max_length=512).input_ids

        prompt_prefix_startidx = (prompt_prefix == self.tokenizer.pad_token_id).int().sum(dim=1)
        B,len_prompt_prefix = prompt_prefix.shape

        prompt_suffix_startidx = (prompt_suffix == self.tokenizer.pad_token_id).int().sum(dim=1)
        prompt_suffix_startidx += len_prompt_prefix
        B, len_prompt_suffix = prompt_suffix.shape

        ans_startidx = (ans == self.tokenizer.pad_token_id).int().sum(dim=1)
        ans_startidx += len_prompt_prefix + len_prompt_suffix

        all_input_ids = torch.cat((prompt_prefix, prompt_suffix, ans), dim=1)

        prompt_embeddings = self.llm_model.get_input_embeddings()(all_input_ids.to(self.llm_model.device))

        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings) #aligned ts embs

        if batch_score is not None:
            flattened_ts = enc_out.contiguous().view(enc_out.shape[0], -1)  # (10, 312*4096)
            vector_norms = torch.norm(flattened_ts, p=2, dim=1, keepdim=True) # 计算每个向量的模长 (L2 范数)

            normalized_flattened_ts = flattened_ts / vector_norms
            similarity_matrix = torch.matmul(normalized_flattened_ts.contiguous(), normalized_flattened_ts.T.contiguous())

            """检查对角线不为1的原因"""
            # diag_elements = similarity_matrix.diag() #不全为1，并不是所有情况都是norm引起的，norm=1的时候，也存在不为1
            # if not torch.allclose(diag_elements, torch.ones_like(diag_elements), atol=1e-6):
            #     print("Warning: Diagonal elements of similarity_matrix are not close to 1.")
            #     print("Diagonal elements:", diag_elements)
            #     vector_norms_after_normalization = torch.norm(normalized_flattened_ts, p=2, dim=1)
            #     print("vector_norms_after_normalization:",vector_norms_after_normalization)  # 有时候等于0.9961

            loss_sim = F.mse_loss(similarity_matrix, batch_score.to(torch.bfloat16))

        prompt_emb_batch = []
        labels_batch = []
        MAX_LENGTH = 0
        for batch_idx in range(B):
            prompt_prefix_each = prompt_embeddings[batch_idx][prompt_prefix_startidx[batch_idx]:len_prompt_prefix]
            ts_each = enc_out[batch_idx]
            # ts_each = torch.zeros_like(ts_each).to(self.llm_model.device)
            prompt_suffix_each = prompt_embeddings[batch_idx][prompt_suffix_startidx[batch_idx]+1:(len_prompt_prefix + len_prompt_suffix)]  #+1，因为第一个token是begintoken，不需要保留

            ans_each = prompt_embeddings[batch_idx][ans_startidx[batch_idx]+1:]
            prompt_embs = torch.cat((prompt_prefix_each, ts_each, prompt_suffix_each, ans_each), dim=0)
            # prompt_embs = torch.cat((prompt_prefix_each, prompt_suffix_each, ans_each), dim=0)
            MAX_LENGTH = max(prompt_embs.shape[0],MAX_LENGTH)
            labels = [-100] * (len(prompt_prefix_each) + len(ts_each) + len(prompt_suffix_each)) + all_input_ids[batch_idx][ans_startidx[batch_idx]+1:].tolist()
            labels_batch.append(labels)
            prompt_emb_batch.append(prompt_embs)

        pad_prompt_embs_batch = []
        pad_attention_mask_batch = []
        pad_labels_batch = []

        pad_token_emb = self.word_embeddings[self.tokenizer.pad_token_id]

        for prompt_emb,labels in zip(prompt_emb_batch,labels_batch):
            l = prompt_emb.shape[0]
            pad_row = pad_token_emb.unsqueeze(0).expand(MAX_LENGTH - l,-1)
            pad_prompt_emb = torch.cat((pad_row, prompt_emb), dim=0)
            # pad_prompt_emb = F.pad(prompt_emb, (MAX_LENGTH - l, 0),value=self.tokenizer.pad_token_id)
            pad_labels = F.pad(torch.tensor(labels), (MAX_LENGTH - len(labels), 0), value=-100)
            pad_attention_mask = F.pad(torch.tensor([1]*l), (MAX_LENGTH - l, 0), value=0)

            pad_prompt_embs_batch.append(pad_prompt_emb)
            pad_attention_mask_batch.append(pad_attention_mask)
            pad_labels_batch.append(pad_labels)

        pad_prompt_embs_batch = torch.stack(pad_prompt_embs_batch).to(self.llm_model.device) # batchsize, MAX_LENGTH, 4096
        pad_attention_mask_batch = torch.stack(pad_attention_mask_batch).to(self.llm_model.device)
        pad_labels_batch = torch.stack(pad_labels_batch).to(self.llm_model.device)

        if batch_score is not None:
            outputs = self.llm_model(inputs_embeds=pad_prompt_embs_batch, attention_mask=pad_attention_mask_batch,
                                     labels=pad_labels_batch)
            loss = 0.5 * outputs.loss + 0.5 * loss_sim #一定要放里面，不然没法成为bfloat、和loss反向传播的数值类型一致
            print(f"total loss:{loss}, similarity loss: {loss_sim}")
            return loss
        else:
            return self.llm_model, pad_prompt_embs_batch, pad_attention_mask_batch, pad_labels_batch, self.tokenizer

    def calcute_lags(self, x_enc): #通过傅里叶变换来提取输入序列的周期性模式，然后计算它们的时序滞后信息，以识别时间序列中最相关的过去时间点
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

# d_model:16 ; n_heads: 8; d_keys:d_ff dim of 全连接层 32 / 128 ; d_llm:model embedding dim 768
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads) #d_keys:d_ff dim of 全连接层 32
        # d_model:时序原始数据经过卷积（局部特征提取）后的维度；是时序被patch分割后，每个patch长度16，再进行一次Conv1D变换，变换后的维度，虽然也被设置为16

        #TODO:查一下多头注意力是不是都是这么写的
        self.query_projection = nn.Linear(d_model, d_keys * n_heads) #时序编码维度16，32（既d_ff dim of 全连接层）*8=256
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads) #768 -> 32*8=256
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm) #256->768
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding): #(32,12,16) (1000,768) (1000,768)
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1) #(32,12,16)->(32,12,32*8=256)->  （32，12，8，32）batchsize32，patchlen12，numhead8，时序编码维度32
        source_embedding = self.key_projection(source_embedding).view(S, H, -1) #(1000,768) -> (1000,32*8=256) ->  (1000，8，32)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1) #同上 1000，8，32

        out = self.reprogramming(target_embedding, source_embedding, value_embedding) #(32,12,8,32)

        out = out.reshape(B, L, -1) #batch，seq_len，head*emb (32,12,8*32=256)

        return self.out_projection(out) #batch，seq_len，d_llm:model embedding dim 768 (32,12,768)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        #TODO:查一下交叉注意力机制代码
        #维度e是求和序列，（共1000个）每个prototype和（共12个patch）每个时序编码求一次分值
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding) #(32,8,12,1000) batch32,numhead8,patchlen12,numtextprototype1000

        A = self.dropout(torch.softmax(scale * scores, dim=-1)) #(32,8,12,1000)
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding) #(32,8,12,1000) (1000,8,32) -> (batch32,patchlen12,numhead8,fullyconnecteddim每个注意力头的维度d_ff 32)

        return reprogramming_embedding #(32,12,8,32)
