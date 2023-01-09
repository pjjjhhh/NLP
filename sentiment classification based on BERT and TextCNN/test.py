import torch
from torch.utils.data import DataLoader
from model import Bert, FC, TextCNN
from dataset import TCDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import argparse
from tqdm import tqdm
import numpy as np
from train import convert_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default='dzdp_data')
    parser.add_argument('-num_workers', type=int, default=2)
    parser.add_argument('-bert_cnn', action='store_true', default=False)
    parser.add_argument('-bert_fc', action='store_true', default=False)
    parser.add_argument('-checkpoint', default=None)
    args = parser.parse_args()

    bert_cnn, bert_fc, checkpoint = args.bert_cnn, args.bert_fc, args.checkpoint
    if bert_cnn == bert_fc:
        raise Exception('should be bert_cnn or bert_fc')
    if not checkpoint:
        raise Exception('checkpoint should not be none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert = Bert().to(device)
    if bert_fc:
        model = FC().to(device)
    elif bert_cnn:
        model = TextCNN(device=device).to(device)

    test_dataset = TCDataset(data_path=args.data_path,
                             test=True)

    test_dataloder = DataLoader(test_dataset,
                                batch_size=64,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=False)

    weights_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights_dict['model_state_dict'])
    if 'bert_state_dict' in weights_dict:
        bert.load_state_dict(weights_dict['bert_state_dict'])

    # test
    bert.eval()
    model.eval()
    outputs = None
    labels = None
    scores = None
    with torch.no_grad():
        for input_ids, masks, _, target in tqdm(test_dataloder, total=len(test_dataloder)):
            input_ids, masks, target = input_ids.to(device), masks.to(device), target.to(device)
            if bert_fc:
                _, bert_out = bert(input_ids, masks)
            elif bert_cnn:
                bert_out, _ = bert(input_ids, masks)
            output = model(bert_out)
            output_arr = output.data.cpu().numpy()
            target_arr = target.data.cpu().numpy()
            score_arr = output_arr[:, 1].copy()  # 取列1，深拷贝
            output_arr = np.argmax(output_arr, axis=1)
            target_arr = np.argmax(target_arr, axis=1)

            if labels is None:
                labels = target_arr
                outputs = output_arr
                scores = score_arr
            else:
                labels = np.concatenate((labels, target_arr), axis=0)
                outputs = np.concatenate((outputs, output_arr), axis=0)
                scores = np.concatenate((scores, score_arr), axis=0)

    acc = accuracy_score(labels, outputs)
    prec, recall, f1, _ = precision_recall_fscore_support(labels, outputs, average='binary')
    auc = roc_auc_score(labels, scores)
    print('\naccuracy:{:.2f}%, precision: {:.2f}, recall: {:.2f}, F1: {:.2f}, AUC:{:.2f}\n'
          .format(100 * acc, 100 * prec, 100 * recall, 100 * f1, 100 * auc))
