import torch
import torch.nn as nn
import transformers
from transformers import BertConfig,ErnieModel,BertTokenizer,ErnieConfig
from component import MeanPooling,criterion,AttentionPooling
from argparse import ArgumentParser
from trainer import train
from utils import set_seed
from tqdm import tqdm
import json
import evaluator
# 定义模型
class UFPErnieModel(nn.Module):
    def __init__(self, pretrained_model,hidden_size=1024, dropout_rate=0.3):
        super(UFPErnieModel, self).__init__()
        self.bert = pretrained_model
        # 全连接层
        # self.fc = nn.Linear(hidden_size, 1)
        # mean_pool+cls
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.mean_pooling = MeanPooling()
        # self.attention_pooling = AttentionPooling(hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, ids, mask, token_type_ids):
        last_hidden_state, pooler_output = self.bert(ids, attention_mask=mask,
                                                     token_type_ids=token_type_ids,
                                                     return_dict=False)

        # output = self.attention_pooling(last_hidden_state,mask)
        output = self.mean_pooling(last_hidden_state, mask)

        out = None
        for i in range(5):
            if i == 0:
                out = self.dropout(output)
                out = self.fc(out)
            else:
                temp_out = self.dropout(output)
                temp_out = self.fc(temp_out)
                out += temp_out
        out = out / 5
        out = self.sigmoid(out)

        # output = torch.cat((mean_output,pooler_output),dim=1)
        # print(output.shape)

        # output = self.dropout(mean_output)
        # output = self.fc(output)
        # output = self.sigmoid(output)

        return out
        # return output

# 加载模型
def load_ernie_model(ernie_model_name,ernie_model_path,device):

    ernie_tokenizer = BertTokenizer.from_pretrained(ernie_model_name)

    ernie_config = ErnieConfig.from_pretrained(ernie_model_name)
    pretrained_model = ErnieModel(config=ernie_config)
    ernie_model = UFPErnieModel(pretrained_model,ernie_config.hidden_size)

    ernie_model.load_state_dict(torch.load(ernie_model_path, map_location=device))

    # for key, v in enumerate(torch.load(ernie_model_path, map_location=device)):
    #     print(key, v.shape)


    ernie_model.to(device)
    print("Load Ernie model success!")

    ernie_model.eval()

    return ernie_model,ernie_tokenizer


@torch.no_grad()
def ernie_inference(ernie_model_name,ernie_model_path,device,test_data_path,mode):
    '''
    生成对测试集的预测文件
    :param ernie_model_name:
    :param ernie_model_path:
    :param device:
    :param test_data_path:
    :return:
    '''
    # 加载模型
    model, tokenizer = load_ernie_model(ernie_model_name,ernie_model_path,device)

    if mode =='test':
        result_file_path =  './predict_data/test/ernie_test_predict.txt'
    else:
        result_file_path = './predict_data/valid/ernie_valid_predict.txt'


    with open(test_data_path, 'r', encoding='utf-8') as fr, \
            open(result_file_path, 'w', encoding='utf-8') as fw:
        for line in tqdm(fr):
            d = json.loads(line)
            import demoji
            query = demoji.replace(d['query'], "")
            replys = d['replys']

            text_pairs = [query + '[SEP]' + demoji.replace(reply['reply'], "") for reply in replys]
            # 当前batch最大文本长度
            max_len = len(max(text_pairs, key=len, default=""))

            inputs = tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=text_pairs,
                truncation=True,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_token_type_ids=True)
            ids = torch.tensor(inputs['input_ids'], dtype=torch.long).to(device)
            mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).to(device)
            token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).to(device)
            output = model(ids, mask, token_type_ids)
            # [bs,1] -> [bs]
            output = output.view(-1)
            # tensor -> list
            output = output.cpu().detach().numpy().tolist()
            # list[float] -> list[str]
            output = [str(pred) for pred in output]

            fw.write('\t'.join(output) + '\n')
    if mode == 'test':
        return result_file_path
    else:
        kl_score = evaluator.evaluation(test_data_path, result_file_path)
        print(kl_score)
        return kl_score




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, default="nghuyong/ernie-3.0-xbase-zh",
                        help="Pretrained Model Name")

    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/ernie/",
                        help="Path or URL of the model")
    parser.add_argument("--train_path", type=str, default="./data/datasets_train.jsonl",
                        help="Path of the train dataset")
    parser.add_argument("--valid_path", type=str, default="./data/datasets_dev.jsonl",
                        help="Path of the valid dataset")

    parser.add_argument("--fold", type=int, default=0, help="5折交叉验证，这个参数用于选择第几折")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of subprocesses for data loading")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--n_accumulate", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--dropout_num", type=int, default=1, help="Muti-sample Dropout")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    # 对抗训练类型
    parser.add_argument('--ad_train',type=str,default="PGD",
                        help = "The type of Adversarial training Function,Default FGM")

    parser.add_argument('--train_augment', action='store_true',
                        help="Train Data Augmentation")

    # MODEL_NAME = 'nghuyong/ernie-3.0-xbase-zh'
    # MODEL_NAME = 'nghuyong/ernie-3.0-base-zh'
    # MODEL_NAME = 'nghuyong/ernie-3.0-nano-zh'

    args = parser.parse_args()
    print(args)
    # 固定随机种子
    set_seed(args.seed)
    # 加载模型
    MODEL_NAME = args.model_name
    pretrained_model = ErnieModel.from_pretrained(MODEL_NAME)
    hidden_size = ErnieConfig.from_pretrained(MODEL_NAME).hidden_size
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained('./tokenizer/' + MODEL_NAME)
    model = UFPErnieModel(pretrained_model, hidden_size, args.dropout_rate)

    model, history = train(args,model,tokenizer,model_name=MODEL_NAME)

    # model.to(args.device)
    # from cross_validation import cross_validation_train
    # best_epoch_klscore = cross_validation_train(args,model,tokenizer,MODEL_NAME)
    # print('Model:{},the Best KLScore of N folds is:{}'.format(MODEL_NAME,best_epoch_klscore))
