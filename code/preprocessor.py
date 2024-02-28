import jsonlines
import pandas as pd
import demoji
from utils import *

def get_rawdata(path):
    SEPCIAL_TOKEN = '[SEP]'
    # 读取数据转成DataFrame格式
    with jsonlines.open(path, 'r') as f:
        data = list()
        for row in f:
            query = row['query']
            replys = row['replys']
            for reply in replys:
                qr_pair = dict()
                # 删除emoji
                repley_content = reply['reply']
                # 删除NaN数据
                if repley_content=='nan':
                    break
                qr_pair['text'] = query + SEPCIAL_TOKEN + demoji.replace(repley_content, "")
                qr_pair['target'] = reply['like'] / (reply['like'] + reply['dislike'])
                data.append(qr_pair)

    return data

def get_dfdata(train_path ='./data/datasets_train.jsonl',valid_path='./data/datasets_dev.jsonl'):
    '''
    读取原始数据->DataFrame
    :param path:
    :return:
    '''
    # other_data = get_other_augment_data()
    train_data = get_rawdata(train_path)
    valid_data = get_rawdata(valid_path)
    augment_data = get_augmentation_data(path='data/augment_train_query_reply_t1.json')
    # augment_data = get_augmentation_data()
    train_augment_data = train_data + augment_data
    # 合并
    train_valid_data = train_data + valid_data

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    train_valid_df = pd.DataFrame(train_valid_data)
    train_augment_df = pd.DataFrame(train_augment_data)

    return train_df,valid_df,train_valid_df,train_augment_df

def get_augmentation_data(path='./data/augmentation_reply_high.json'):
    data = load_json2data(path)

    print(len(data))
    return data

def get_valid_augment_data(path='./data/augmentation_reply_valid_data.json'):
    high = list()
    data = load_json2data(path)
    for d in data:
        if d['target'] == 1:
            del d['strategy']
            high.append(d)

    return high

def get_other_augment_data(path='./data/augmentation_other_data.json'):
    data = load_json2data(path)
    for d in data:
        del d['strategy']
    return data

def data_analysis(train_valid_df):
    count = dict()
    count['zero'] = 0
    count['low'] = 0
    count['mid'] = 0
    count['high'] = 0
    count['highest'] = 0
    target = train_valid_df['target'].values.tolist()
    for t in target:
        if t==0:
            count['zero'] +=1
        elif t<=0.3 and t > 0:
            count['low'] += 1
        elif t>0.3 and t<0.6:
            count['mid'] += 1
        elif t<1 and t>=0.6:
            count['high'] += 1
        else:
            count['highest'] += 1
    # {'zero': 1470, 'low': 5859, 'mid': 12166, 'high': 24022, 'highest': 6586}
    # 扩充zero low 和higest的数据
    # 扩充mid和highest的数据
    print(count)

def genetate_cv_data(train_path ='./data/datasets_train.jsonl',valid_path='./data/datasets_dev.jsonl'):
    '''
       读取原始数据->DataFrame
       :param path:
       :return:
       '''
    # other_data = get_other_augment_data()
    train_data = get_rawdata(train_path)
    valid_data = get_rawdata(valid_path)
    train_high_augment_data = get_augmentation_data(path='./data/augment_train_query_reply_t1.json')
    valid_high_augment_data = get_valid_augment_data(path='./data/augment_dev_query_reply_t1.json')
    print(train_high_augment_data[2:10])
    print(valid_high_augment_data[2:10])
    data = train_data + valid_data + train_high_augment_data + valid_high_augment_data

    data = pd.DataFrame(data)

    from inputter import create_folds
    df = create_folds(data, 5, 42, n_grp=10)
    text_length = df['text'].apply(lambda x: len(x))
    max_len = max(text_length)
    print(df.head)
    print(max_len)

    for fold in range(5):
        fold_train_data = list()
        fold_valid_data = list()
        train_df = df[df.kfold != fold].reset_index(drop=True)
        valid_df = df[df.kfold == fold].reset_index(drop=True)
        for index,row in train_df.iterrows():
            d = dict()
            text = row['text']
            target = row['target']
            d['text'] = text
            d['target'] = target
            fold_train_data.append(d)
        for index,row in valid_df.iterrows():
            d = dict()
            text = row['text']
            target = row['target']
            d['text'] = text
            d['target'] = target
            fold_valid_data.append(d)
        save_data2json('./cv_data_v2/{}fold_train_data.json'.format(fold),fold_train_data)
        save_data2json('./cv_data_v2/{}fold_dev_data.json'.format(fold),fold_valid_data)



if __name__ == '__main__':
    # train_df, valid_df, train_valid_df, train_augment_df = get_dfdata()
    # print(len(train_augment_df))
    # data = load_json2data('./data/augment_train_query_reply_t1.json')
    # print(len(data))
    # genetate_cv_data()
    # train_data = get_rawdata('./data/datasets_train.jsonl')
    # valid_data = get_rawdata('./data/datasets_dev.jsonl')
    # train_high_augment_data = get_augmentation_data(path='./data/augmentation_reply_high.json')
    # valid_high_augment_data = get_valid_augment_data()
    #
    # data = train_data + valid_data + train_high_augment_data + valid_high_augment_data
    # print(len(data))
    total = 0
    for i in range(5):
        fold_data =load_json2data('./cv_data_v2/{}fold_dev_data.json'.format(i))
        c = len(fold_data)
        total = c + total
    print(total)