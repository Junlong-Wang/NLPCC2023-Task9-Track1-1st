from run_bert import bert_inference
from run_ernie import ernie_inference
import numpy as np



def model_inference(args):
    # 执行预测
    # valid模式返回得是
    ernie_result = ernie_inference(ernie_model_name='nghuyong/ernie-3.0-xbase-zh',
                                   ernie_model_path='./checkpoint/ernie/91.58297_e3_ernie-3.0-xbase-zh.bin',
                                   device='cuda:0',
                                   test_data_path=test_data_path,
                                   mode=args.mode)
    roberta_result = bert_inference(bert_model_name='hfl/chinese-roberta-wwm-ext-large',
                                    bert_model_path='checkpoint/roberta-large/91.47269_e2_chinese-roberta-wwm-ext-large.bin',
                                    device='cuda:0',
                                    test_data_path=test_data_path,
                                    mode=args.mode
                                    )
    macbert_result = bert_inference(bert_model_name='hfl/chinese-macbert-large',
                                    bert_model_path='./checkpoint/macbert/91.38763_e4_chinese-macbert-large.bin',
                                    device='cuda:0',
                                    test_data_path=test_data_path,
                                    mode=args.mode
                                    )


def simple_combine_inference(mode='valid'):
    '''
    传入多个模型预测的结果
    :param model:
    :return:
    '''
    # 两种模型的预测数据，后面可以再扩充
    if mode == 'valid':
        ernie_test_result = './predict_data/valid/ernie_valid_predict.txt'
        roberta_test_result = './predict_data/valid/chinese-roberta-wwm-ext-large_valid_predict.txt'
        macbert_test_result = './predict_data/valid/chinese-macbert-large_valid_predict.txt'
        combine_test_result = './predict_data/valid/simple_combine_valid_predict.txt'
    else:
        ernie_test_result = './predict_data/test/ernie_test_predict.txt'
        roberta_test_result = './predict_data/test/chinese-roberta-wwm-ext-large_test_predict.txt'
        macbert_test_result = './predict_data/test/chinese-macbert-large_test_predict.txt'
        combine_test_result = './predict_data/test/simple_combine_valid_predict.txt'

    combinef = open(combine_test_result, 'w', encoding='utf-8')

    ernief = open(ernie_test_result, 'r', encoding='utf-8')
    robertaf = open(roberta_test_result, 'r', encoding='utf-8')
    macbertf = open(macbert_test_result, 'r', encoding='utf-8')

    for erniex, robertax, macbertx in zip(ernief, robertaf, macbertf):
        # 读取一行
        ernie_pred = [float(x) for x in erniex.split()]
        roberta_pred = [float(x) for x in robertax.split()]
        macbert_pred = [float(x) for x in macbertx.split()]

        # combine_pred = [e*0.56+r*0.44 for e, r, m in zip(ernie_pred, roberta_pred, macbert_pred)]

        combine_pred = [e*0.43+r*0.3+m*0.27  for e, r, m in zip(ernie_pred, roberta_pred, macbert_pred)]

        pred = [str(p) for p in combine_pred]
        combinef.write('\t'.join(pred) + '\n')

    ernief.close()
    robertaf.close()
    macbertf.close()

    if mode == 'valid':
        from evaluator import evaluation
        score = evaluation('./data/datasets_dev.jsonl', combine_test_result)
        print(score)
        combinef.close()


if __name__ == '__main__':



    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="valid",
                        help="预测类型，是在验证集上验证融合的结果，还是直接在测试集上推理")

    args = parser.parse_args()
    print(args)

    if args.mode == "test":
        test_data_path = './data/datasets_test_track1.jsonl'
    else:
        test_data_path = './data/datasets_dev.jsonl'

    # model_inference(args)

    simple_combine_inference(mode=args.mode)

    # 执行预测






