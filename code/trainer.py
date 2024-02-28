from argparse import ArgumentParser
import gc
import copy
import time
import json
import torch
from torch.cuda import amp
from transformers import BertTokenizer,BertConfig,RobertaConfig,RobertaTokenizer,AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from inputter import *
from component import *
from colorama import Fore
b_ = Fore.BLUE
import os
import warnings
import numpy as np
from utils import *
warnings.filterwarnings("ignore")
from component import criterion


def train_one_epoch(model, args, optimizer, scheduler, attacker, dataloader, epoch):

    model.train()
    # 自动混合精度
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:

        ids = data['ids'].to(args.device, dtype=torch.long)
        mask = data['mask'].to(args.device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(args.device, dtype=torch.long)
        targets = data['target'].to(args.device, dtype=torch.float)

        batch_size = ids.size(0)

        # 正常训练
        with amp.autocast(enabled=True):
            outputs = model(ids, mask, token_type_ids)
            loss = criterion(outputs, targets)
            loss = loss / args.n_accumulate
        # 反向传播得到正常的grad
        scaler.scale(loss).backward()

        #----------------对抗训练----------------------#
        #PGD对抗方法

        if args.ad_train=='FGM':

            attacker.attack()

            with amp.autocast(enabled=True):
                outputs_adv = model(ids, mask, token_type_ids)
                loss_adv = criterion(outputs_adv, targets)
                loss_adv = loss_adv / args.n_accumulate

            scaler.scale(loss_adv).backward()
            attacker.restore()
        # PGD对抗方法
        elif args.ad_train=='PGD':
            # 这行代码别忘了
            attacker.backup_grad()

            k = 3
            for t in range(k):
                # 在embedding上添加对抗扰动, first attack时备份param.data
                attacker.attack(is_first_attack=(t == 0))
                if t != k - 1:
                    model.zero_grad()
                else:
                    attacker.restore_grad()
                with amp.autocast(enabled=True):
                    outputs_adv = model(ids, mask, token_type_ids)
                    loss_adv = criterion(outputs_adv, targets)
                    loss_adv = loss_adv / args.n_accumulate
                    # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    scaler.scale(loss_adv).backward()

            # 恢复embedding参数
            attacker.restore()



        if (step + 1) % args.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,KFold=args.fold,Model=args.model_name,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model,model_name, optimizer, tokenizer, dataloader, device,valid_path, epoch,fold):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    TARGETS = []
    PREDS = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.float)

        batch_size = ids.size(0)

        outputs = model(ids, mask, token_type_ids)

        loss = criterion(outputs, targets)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        PREDS.extend(outputs.cpu().detach().numpy().tolist())
        TARGETS.extend(targets.cpu().detach().numpy().tolist())

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,KFold=fold,Model=model_name,
                        LR=optimizer.param_groups[0]['lr'])

    val_rmse = mean_squared_error(TARGETS, PREDS, squared=False)

    val_klscore = inference(model,tokenizer,device,valid_path,model_name)

    gc.collect()

    return epoch_loss, val_rmse,val_klscore







@torch.no_grad()
def inference(model,tokenizer, device,valid_path,model_name):


    result_file_path = './predict_data/valid/'+model_name.split('/')[-1]+'_predict.txt'

    with open(valid_path, 'r', encoding='utf-8') as fr, \
            open(result_file_path, 'w', encoding='utf-8') as fw:
        for line in tqdm(fr):
            d = json.loads(line)
            import demoji
            query = demoji.replace(d['query'], "")
            replys = d['replys']

            text_pairs = [query + '[SEP]' + demoji.replace(reply['reply'], "") for reply in replys]
            # 当前batch最大文本长度
            max_len = len(max(text_pairs,key=len,default=""))

            inputs = tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=text_pairs,
                truncation=True,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                return_token_type_ids=True)

            ids = torch.tensor(inputs['input_ids'],dtype=torch.long).to(device)
            mask = torch.tensor(inputs['attention_mask'],dtype=torch.long).to(device)
            token_type_ids = torch.tensor(inputs["token_type_ids"],dtype=torch.long).to(device)

            output = model(ids,mask,token_type_ids)

            # [bs,1] -> [bs]
            output = output.view(-1)
            # tensor -> list
            output = output.cpu().detach().numpy().tolist()
            # list[float] -> list[str]
            output = [str(pred) for pred in output]

            fw.write('\t'.join(output) + '\n')
    from evaluator import evaluation
    score = evaluation(valid_path,result_file_path)

    gc.collect()
    return score



def train(args,model,tokenizer,model_name):

    model.to(args.device)

    train_loader, valid_loader = prepare_cv_data(args,tokenizer)
    print("Len of train loader:", len(train_loader))
    print("Len of valid loader:",len(valid_loader))
    # df,max_len = prepare_nfold_data(args)
    # train_loader, valid_loader = get_dataloader(df,fold=0,tokenizer=tokenizer,max_len=max_len,args=args)

    # Defining Optimizer with weight decay to params other than bias and layer norms
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    attacker = None

    if args.ad_train == "FGM":
        print('------Adversarial Training Method is {}------'.format(args.ad_train))
        attacker = FGM(model, epsilon=1, emb_name='embeddings.word_embeddings')
    elif args.ad_train == "PGD":
        print('------Adversarial Training Method is {}------'.format(args.ad_train))
        attacker = PGD(model,emb_name='embeddings.word_embeddings')

    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate)
    # Defining LR Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * args.n_epochs
    )


    start = time.time()
    # 初始化
    best_model_wts = copy.deepcopy(model.state_dict())
    # 初始化为负无穷
    best_epoch_klscore = -np.inf
    # best_epoch_rmse = np.inf
    history = defaultdict(list)

    for epoch in range(1, args.n_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, args, optimizer, scheduler, attacker, dataloader=train_loader, epoch=epoch)

        valid_epoch_loss, valid_epoch_rmse,val_epoch_klscore = valid_one_epoch(
                                                             model= model,model_name= model_name,
                                                             optimizer=optimizer, tokenizer=tokenizer,
                                                             dataloader=valid_loader,
                                                             device=args.device,valid_path=args.valid_path,epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(valid_epoch_loss)
        history['Valid RMSE'].append(valid_epoch_rmse)
        history['Valid KLScore'].append(val_epoch_klscore)
        print(f'Valid RMSE: {valid_epoch_rmse}')
        print(f'Valid KLScore: {val_epoch_klscore}')
        # deep copy the model
        if val_epoch_klscore >= best_epoch_klscore:
            print(f"{b_}Validation KLScore Improved ({best_epoch_klscore} ---> {val_epoch_klscore})")
            best_epoch_klscore = val_epoch_klscore
            # best_epoch_rmse = valid_epoch_rmse
            best_model_wts = copy.deepcopy(model.state_dict())
            # 保存模型
            PATH = os.path.join(args.checkpoint_path,
                                "{:.5f}_{}f_e{:.0f}_{}.bin".format(best_epoch_klscore,args.fold,epoch,
                                                               model_name.split('/')[-1])) # hfl/chinese-bert-wwm-ext
            torch.save(model.state_dict(), PATH)
            print("Model Saved")

        # print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    # print("Best Loss: {:.4f}".format(best_epoch_rmse))
    print("Best KLScore: {:.5f}".format(best_epoch_klscore))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


if __name__ == '__main__':
    pass




