from utils import save_data2json
import random
from preprocessor import *
from tqdm import tqdm
import time
from nlpcda import Similarword,RandomDeleteChar,CharPositionExchange,baidu_translate
train_data = get_rawdata('./data/datasets_train.jsonl')
valid_data = get_rawdata('./data/datasets_dev.jsonl')
# 合并
train_valid_data = train_data + valid_data


# 随机同义词替换
def replace_simword(reply):
    smw  = Similarword(create_num=3, change_rate=0.5)
    pseudo_replys = smw.replace(reply)
    for pseudo_reply in pseudo_replys:
        if pseudo_reply!=reply:
            return pseudo_reply
    return ""

# 随机删除
def delete_randomword(reply):
    smw = RandomDeleteChar(create_num=3, change_rate=0.3)
    pseudo_replys = smw.replace(reply)
    for pseudo_reply in pseudo_replys:
        if pseudo_reply != reply:
            return pseudo_reply
    return ""
# 随机置换临近字
def exchange_position(reply):
    smw = CharPositionExchange(create_num=3, change_rate=0.3, char_gram=3, seed=42)
    pseudo_replys = smw.replace(reply)
    for pseudo_reply in pseudo_replys:
        if pseudo_reply != reply:
            return pseudo_reply
    return ""

# 回译法
def back_translate(reply):
    try:
        en_reply = baidu_translate(content=reply, appid='20230526001690189', secretKey='n_AZizmkeKkExLysOLZx', t_from='zh',
                               t_to='en')
        time.sleep(1)
        pseudo_reply = baidu_translate(content=en_reply, appid='20230526001690189', secretKey='n_AZizmkeKkExLysOLZx', t_from='en',
                               t_to='zh')

        # 翻译结果和原来一样
        if pseudo_reply == reply:
            return ""
        else:
            return pseudo_reply
    except:
        print('======回译法异常！======')
        return ""

def get_other_data():
    other = list()
    for data in train_data:
        target = data['target']
        if target!=1:
           other.append((data['text'],data['target']))
    return other

def get_validdata():
    valid = list()
    for data in valid_data:
        valid.append((data['text'],data['target']))
    return valid


def get_highdata():
    high = list()
    for data in train_data:
        target = data['target']
        if target==1:
           high.append((data['text'],data['target']))
    return high

def data_augmentation_reply(data):
    random.seed(42)

    result = list()

    for text,target in tqdm(data):

        text = text.split('[SEP]')
        # print(text)
        query = text[0]
        reply = text[1]
        # 拿Reply去做数据增强
        # [0,4)
        choice = random.randint(0, 4)
        strategy = ""

        if choice == 0:
            pseudo_reply = back_translate(reply)
            strategy = "BackTranslate"
            time.sleep(1)
        elif choice == 1:
            pseudo_reply = replace_simword(reply)
            strategy = "ReplaceSimilarWord"
        elif choice == 2:
            pseudo_reply = delete_randomword(reply)
            strategy = "RandomDeleteChar"
        elif choice == 3:
            pseudo_reply = exchange_position(reply)
            strategy = "CharPositionExchange"
        else:
            pseudo_reply = ""

        # 翻译结果不能和原来一样
        if pseudo_reply=="":
            continue

        elif pseudo_reply!="" and pseudo_reply!= reply:
            pseudo_pair = dict()
            pseudo_pair['text'] = str(query)+'[SEP]'+str(pseudo_reply)
            pseudo_pair['target'] = target
            pseudo_pair['strategy'] = strategy
            result.append(pseudo_pair)

    return result


def data_augmentation_query_reply(data):
    random.seed(42)

    result = list()

    for text,target in tqdm(data):

        text = text.split('[SEP]')
        # print(text)
        query = text[0]
        reply = text[1]

        # 拿query去做数据增强，初始化
        tolerance = 0
        pseudo_query = ""
        query_strategy = ""
        while pseudo_query=="" and tolerance <= 10:
            # 必须增强，不能是空，除非增强了10次都没结果
            pseudo_query,query_strategy = single_augment(query)
            tolerance += 1
        time.sleep(1)
        # 拿reply去做数据增强，初始化
        tolerance = 0
        pseudo_reply = ""
        reply_strategy = ""
        while pseudo_reply == "" and tolerance <= 10:
            # 必须增强，不能是空，除非增强了10次都没结果
            pseudo_reply,reply_strategy = single_augment(reply)
            tolerance += 1

        if pseudo_reply!="" and pseudo_reply!= reply and pseudo_query!="" and pseudo_query!=query:
            pseudo_pair = dict()
            pseudo_pair['text'] = str(pseudo_query)+'[SEP]'+str(pseudo_reply)
            pseudo_pair['target'] = target
            pseudo_pair['strategy'] = query_strategy+','+reply_strategy
            result.append(pseudo_pair)

    return result


# 单条Query或者Reply做增强
def single_augment(sample):
    choice = random.randint(0, 4)
    strategy = ""
    if choice == 0:
        pseudo_sample = back_translate(sample)
        strategy = "BackTranslate"
        # time.sleep(1)
    elif choice == 1:
        pseudo_sample = replace_simword(sample)
        strategy = "ReplaceSimilarWord"
    elif choice == 2:
        pseudo_sample = delete_randomword(sample)
        strategy = "RandomDeleteChar"
    elif choice == 3:
        pseudo_sample = exchange_position(sample)
        strategy = "CharPositionExchange"
    else:
        pseudo_sample = ""

    return pseudo_sample,strategy



if __name__ == '__main__':
    high = get_highdata()
    result = data_augmentation_query_reply(high)
    save_data2json('./augment_train_query_reply_t1.json',result)







