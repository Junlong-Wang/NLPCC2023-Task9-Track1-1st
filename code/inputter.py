
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from preprocessor import get_dfdata


# N折交叉验证
def create_folds(df, n_folds, seed, n_grp=None):
    # 直接添加一列
    df['kfold'] = -1

    # 标准交叉验证划分呢
    if n_grp is None:
        skf = KFold(n_splits=n_folds, random_state=seed)
        target = df.target
    else:
        # 分层交叉验证，确保每一折的标签分布和原始数据相同
        skf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=seed)
        # 添加一列，按区间划分连续值target
        df['grp'] = pd.cut(df.target, n_grp, labels=False)
        target = df.grp

    # skf.split(target.index,target.values)
    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        # 标记该行数据作为第几折的验证集数据
        df.loc[v, 'kfold'] = fold_no
    return df



# 创建数据集 UserFeedbackPrediction
class UFPDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.text = df['text'].values
        self.target = df['target'].values
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(self.target[index], dtype=torch.float)
        }


def prepare_data(args,tokenizer):

    train_df, valid_df, train_valid_df,train_augment_df = get_dfdata(args.train_path, args.valid_path)


    text_lenghts = train_df['text'].apply(lambda x: len(x))
    max_len = max(text_lenghts)
    # 扩充训练数据，数据增强
    if args.train_augment:
        train_dataset = UFPDataset(train_augment_df, tokenizer, max_len)
    else:
        train_dataset = UFPDataset(train_df, tokenizer, max_len)

    valid_dataset = UFPDataset(valid_df, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False, pin_memory=True)
    return train_loader,valid_loader

#----------------------------------------------------#


def prepare_nfold_data(args):
    '''
    把训练集和验证集合并，构建含有N-折交叉验证的数据，
    :param args:
    :return:DataFrame类型数据和文本的最大长度
    '''
    # 原始数据->DataFrame
    train_df, valid_df, train_valid_df, train_augment_df = get_dfdata(args.train_path,args.valid_path)
    # N折交叉验证
    df = create_folds(train_valid_df, args.n_folds, args.seed, n_grp=10)
    text_length = df['text'].apply(lambda x: len(x))
    max_len = max(text_length)
    print(max_len)
    return df,max_len

def get_dataloader(df,fold,tokenizer,max_len,args):
    '''
    :param df:
    :param max_len:
    :param fold:n_fold折交叉验证，共有n_fold，取第fold份数据
    :param args:
    :return:
    '''
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)





    train_dataset = UFPDataset(train_df, tokenizer, max_len)
    valid_dataset = UFPDataset(valid_df, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False, pin_memory=True)

    return train_loader,valid_loader

#---------------------------------------------------------------------------#
def prepare_cv_data(args,tokenizer):

    from utils import load_json2data

    train_data = load_json2data('./cv_data_v2/{}fold_train_data.json'.format(args.fold))
    valid_data = load_json2data('./cv_data_v2/{}fold_dev_data.json'.format(args.fold))
    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    print('======You are using the #{}# fold data to train!======'.format(args.fold))
    text_lenghts = train_df['text'].apply(lambda x: len(x))
    max_len = max(text_lenghts)

    train_dataset = UFPDataset(train_df, tokenizer, max_len)

    valid_dataset = UFPDataset(valid_df, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False, pin_memory=True)
    return train_loader, valid_loader

