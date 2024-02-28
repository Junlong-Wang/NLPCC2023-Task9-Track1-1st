import jsonlines
import torch
import datetime
import json
import numpy as np
import os

# 设置随机种子
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)



# 对抗训练
class FGM:
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                # print(param.grad)
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# PGD
class PGD:
    def __init__(self, model,emb_name, eps=1, alpha=0.3):
        self.model = model
        self.emd_name = emb_name
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self,is_first_attack=True):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emd_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emd_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]










# 实验记录
def record(model_name,learning_rate,dropout_rate,batch_size,n_accumualte,RMSE,KLScore):
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('./record.txt', 'a') as f:
        f.write('=======================================\n')
        f.write('time:{}\n'.format(nowtime))
        f.write('model:{}\n'.format(model_name))
        f.write('learning_rate:{}\n'.format(learning_rate))
        f.write('droupout_rate:{}\n'.format(dropout_rate))
        f.write('batch_size:{}\n'.format(batch_size))
        f.write('n_accumualte:{}\n'.format(n_accumualte))
        f.write('RMSE:{}\n'.format(RMSE))
        f.write('KLScore:{}\n'.format(KLScore))
        f.write('=======================================\n')


# 文件读写
def save_data2json(newfile_path,data):
    '''
    把数据保存为json文件
    :param newfile_path:
    :param data:
    :return:
    '''
    data = json.dumps(data,ensure_ascii=False)
    with open(newfile_path, 'w', encoding='utf-8') as f:
        f.write(data)


def load_json2data(file_path):
    '''
    把数据从json中读取
    :param file_path:
    :return:
    '''
    with open(file_path,mode='r',encoding='utf-8') as f:
        data = json.load(f)
    return data



def json2jsonl(oldfile_path,newfile_path):
    '''
    json文件转换为jsonl文件格式
    :param oldfile_path:
    :param newfile_path:
    :return:
    '''
    data = load_json2data(file_path=oldfile_path)
    with jsonlines.open(newfile_path, "w") as writer:
        for d in data:
            writer.write(d)

